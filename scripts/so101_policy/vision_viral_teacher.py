"""Markov joint teacher (JT) of build9 (``nedm.so101_policy.joint_env`` / ``joint_teacher``; docs
~/so101_policy/docs/build9_viral_student.md).

    export PYTHONPATH=src PY=~/miniconda3/envs/so101_vision/bin/python
    $PY -P scripts/so101_policy/vision_viral_teacher.py tasks --kind C --n 100 --dev --out dev100.npz   # MuJoCo tasks
    $PY -P scripts/so101_policy/vision_viral_teacher.py expert --n 8192 --noise-frac 0.5 --noise 1.0 --out e0.npz
    $PY -P scripts/so101_policy/vision_viral_teacher.py bc --data e0.npz [e1.npz ...] --out RUN/bc
    $PY -P scripts/so101_policy/vision_viral_teacher.py eval --jt RUN/bc/bc.pt --tasks dev100.npz --world mujoco
    $PY -P scripts/so101_policy/vision_viral_teacher.py eval-expert --tasks dev100.npz --world mujoco   # current teacher
    $PY -P scripts/so101_policy/vision_viral_teacher.py ppo --init RUN/bc/bc.pt --out RUN/ppo --iterations 200

Task kinds: ``C`` = build8's stage-C sampler (collector start sampler + MuJoCo warm-up; random cube, random goal 6-30 cm,
yaw +-30 deg), seeds 640,000+ (training) or 590,000+ (``--dev``; ``--first 1000`` = the fresh set 591,000+, ``--first 2000`` videos);
``bank`` = start bank + sample_goals (NRD only), seeds 700,000+; ``H`` (real2sim) = HOME starts (``tray.make_tasks``
stage H: the real robot's HOME pose + ``--home-config`` noise, a start-bank cube, stage C's goal), seeds 690,000+
(training) or 597,000+ (``--dev``; ``--first 1000`` = fresh 598,000+). ``mix`` = a training table of several task files
(e.g. 50 % C + 50 % H, rows shuffled):

    $PY -P scripts/so101_policy/vision_viral_teacher.py tasks --kind H --n 100 --dev --home-config \
        configs/so101_policy/vision/viral_r2s_home.json --out home_dev100.npz
    $PY -P scripts/so101_policy/vision_viral_teacher.py mix --inputs C.npz H.npz --counts 8192 8192 --out mix.npz

``starts`` = a PPO start bank from start banks and training task files (no goals, no seeds; e.g. the round v8 roll-cap
fine-tune's bank: bank_pp_split0 once + the HOME45 training starts 3 times, ~50 % HOME45):

    $PY -P scripts/so101_policy/vision_viral_teacher.py starts --inputs bank_pp_split0.npz train_H45_8k.npz \
        --repeat 1 3 --exclude dev100.npz fresh100.npz home45_dev100.npz home45_fresh100.npz --roll-hi-deg 98 \
        --out bank_pp_h45mix.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import joint_teacher as JT

TEACHER = "~/so101_policy/runs/c1ppo/ckpt/nC1_s62_b_model_00171.pt"
NRD = "~/so101_policy/ckpt/nrd_c1_s62.pt"
MJ_PY = "~/miniconda3/envs/nrdsim/bin/python"
TRAIN_FIRST, DEV_FIRST = 90_000, 28_000      # tray.make_tasks("C", first=...): 640,000+ (training) / 590,000+ (dev)


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def x(p):
    return str(Path(p).expanduser()) if p else p


def cmd_tasks(a):
    if a.kind == "bank":
        t = JT.bank_tasks(a.n, a.seed, a.first)
    elif a.kind == "H":
        from nedm.so101_policy.vision import tray as TR
        home = json.loads(Path(a.home_config).read_text()) if a.home_config else {}
        t = TR.make_tasks("H", a.n, a.first, a.dev, log=log, home=home)
    else:
        from nedm.so101_policy.vision import tray as TR
        t = TR.make_tasks("C", a.n, (DEV_FIRST if a.dev else TRAIN_FIRST) + a.first, a.dev, log=log)
    from nedm.so101_policy.vec_env import reserved_seed_hits
    if reserved_seed_hits(t["seeds"]):
        raise ValueError("reserved seeds")
    np.savez(a.out, **t)
    log(f"-> {a.out}: {len(t['seeds'])} tasks, seeds {t['seeds'][0]}-{t['seeds'][-1]}")


def cmd_mix(a):
    from nedm.so101_policy.vision import tray as TR
    from nedm.so101_policy.vec_env import reserved_seed_hits
    tables = [dict(np.load(x(p_))) for p_ in a.inputs]
    t = TR.mix_tasks(tables, a.counts, a.seed)
    if reserved_seed_hits(t["seeds"]):
        raise ValueError("reserved seeds")
    t["mix_inputs"] = np.array([str(x(p_)) for p_ in a.inputs])
    np.savez(a.out, **t)
    log(f"-> {a.out}: {len(t['seeds'])} tasks, kinds {np.bincount(t['kind']).tolist()} ({t['kind_stage'].tolist()})")


def cmd_starts(a):
    """A PPO start bank (``vec_env.load_starts`` layout: q, qd, cube_pos, cube_quat, cube_v, cube_w; no goals, no seeds)
    from start banks and TRAINING task files (their ``state_*`` rows): every row of input i ``--repeat[i]`` times, in
    input order (the env draws rows uniformly, so the repeats set the mix). Refused: a sealed-test path, a bank with a
    reserved seed, a task file with a reserved seed or ``dev`` set, a task file sharing a seed with an ``--exclude``
    file (development / fresh sets). Rows above ``--roll-hi-deg`` are dropped (``JE.drop_high_roll``)."""
    import math
    from nedm.so101_policy.vec_env import reserved_seed_hits
    rep = list(a.repeat or [1]) * (len(a.inputs) if len(a.repeat or [1]) == 1 else 1)
    if len(rep) != len(a.inputs) or min(rep) < 1:
        raise SystemExit("--repeat: one value (>= 1) or one per input")
    ex = set()
    for p_ in a.exclude or []:
        ex |= set(np.asarray(np.load(x(p_))["seeds"]).tolist())
    keys = ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")
    parts, src, rows = [], [], []
    for i, (p_, k) in enumerate(zip(a.inputs, rep)):
        if "test_sealed" in str(p_):
            raise SystemExit(f"{p_}: the sealed test data must not be used for starts")
        f = np.load(x(p_))
        task = "goal_xy" in f.files
        sd = [np.asarray(f[n]) for n in ("seeds", "seed") if n in f.files]
        hits = sum(reserved_seed_hits(v) for v in sd)
        if hits:
            raise SystemExit(f"{p_}: {hits} reserved (validation / test) seeds")
        if task:
            if "dev" in f.files and bool(f["dev"]):
                raise SystemExit(f"{p_}: a development task set")
            if not sd:
                raise SystemExit(f"{p_}: a task file without seeds")
            ov = ex & set(sd[0].tolist())
            if ov:
                raise SystemExit(f"{p_}: {len(ov)} seeds are in the --exclude sets")
        pre = "state_" if task else ""
        d = {n: np.asarray(f[pre + n], np.float64) for n in keys}
        m = len(d["q"])
        parts.append({n: np.concatenate([v] * k) for n, v in d.items()})
        src.append(np.full(m * k, i, np.int16))
        rows.append(np.tile(np.arange(m), k))
        log(f"{p_}: {m} rows x {k} ({'task file' if task else 'bank'})")
    out = {n: np.concatenate([q[n] for q in parts]) for n in keys}
    out["src_input"], out["src_row"] = np.concatenate(src), np.concatenate(rows)
    if a.roll_hi_deg is not None:
        t, n_drop = JE.drop_high_roll({n: torch.as_tensor(v) for n, v in out.items()},
                                      math.radians(JE.roll_hi_config(a.roll_hi_deg)))
        out = {n: v.numpy() for n, v in t.items()}
        log(f"roll cap {a.roll_hi_deg:g} deg: {n_drop} rows dropped")
    out["src_inputs"] = np.array([str(x(p_)) for p_ in a.inputs])
    out["src_repeat"] = np.asarray(rep, np.int64)
    np.savez(a.out, **out)
    sh = np.bincount(out["src_input"], minlength=len(a.inputs)) / len(out["q"])
    log(f"-> {a.out}: {len(out['q'])} rows, share per input {np.round(sh, 4).tolist()}, wrist roll max "
        f"{math.degrees(float(out['q'][:, JE.ROLL].max())):.1f} deg")


def cmd_expert(a):
    from nedm.so101_policy.vision import teacher as TT
    te = TT.Teacher(x(a.teacher))
    if a.tasks:
        t = dict(np.load(a.tasks))
    else:
        t = JT.bank_tasks(a.n, a.seed, a.first)
    n = len(t["seeds"])
    nz = np.zeros(n)
    nz[int(round(n * (1 - a.noise_frac))):] = a.noise
    d = JT.expert_collect(t, te, a.world, nz, a.seed, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon,
                          a.batch, log, record=not a.no_record)
    np.savez_compressed(a.out, **d)
    log(f"-> {a.out}: success {np.mean(d['success']) * 100:.1f} %, lenient {np.mean(d['lenient']) * 100:.1f} %")


def cmd_bc(a):
    data = JT.BCSet([x(p) for p in a.data], a.device, seed=a.seed)
    log(f"bc data: {len(data.train_rows)} train rows, {len(data.val_rows)} val rows")
    env_cfg = dict(JE.JointEnvConfig(ckpts=[x(a.nrd_ckpt)]).to_dict())
    JT.train_bc(data, a.out, tuple(a.hidden), a.steps, a.batch, a.lr, seed=a.seed, log=log, env_cfg=env_cfg,
                rate_noise=a.rate_noise)
    (Path(a.out) / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=1))


def load_tasks(path, n=0, first=0):
    from nedm.so101_policy.vision.demos import load_tasks as lt
    return lt(path, np.arange(first, first + n) if n else None)


def cmd_eval(a):
    jt = JE.JointTeacher(x(a.jt), a.device)
    t = load_tasks(a.tasks, a.n, a.first)
    B = len(t["seeds"])
    env = JT.make_env(a.world, B, jt.codec, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon,
                      lead_clamp=not a.no_lead, floor_guard=not a.no_floor, hold_cap_s=a.hold_cap_s,
                      hold_obs_s=a.hold_obs_s, roll_hi_deg=a.roll_hi_deg)
    if env.dyn is not None:       # $NEDM_SO101_DYN (joint_teacher.make_env)
        log(f"robot dynamics ({a.world}, {B} worlds): {json.dumps(env.dyn_info()['cfg'])}")
    if env.roll_hi is not None:   # the wrist-roll cap (--roll-hi-deg)
        log(f"roll cap: {json.dumps(env.roll_info())}")
    from nedm.so101_policy import table as TB
    plane, floor, _, moved = TB.from_cli(a)
    if moved:      # evaluation only: a lowered / tilted table (table.py)
        log(f"table: {TB.describe(plane)}, floor {floor}: {json.dumps(env.set_table(plane, floor))}")
    extra = TB.log_only_extra(env, a)
    if extra:
        log(f"log-only task rules: {env.tcfg.log_only}")
    r = JT.run_tasks(env, t, jt, noise=a.noise, record_states=a.record_states, log=log)
    env.close()
    if moved:
        r["table"] = np.array(json.dumps(env.table_info()))
    if extra:
        r["log_only"] = np.array(json.dumps(list(env.tcfg.log_only)))
    s = JT.summarize(r)
    if env.dyn is not None:       # robot dynamics of the evaluation ($NEDM_SO101_DYN, joint_teacher.make_env)
        r["dyn"] = np.array(json.dumps(env.dyn_info()))
        on = env.dyn.params()[:, 0].cpu().numpy() > 0.5
        le = np.asarray(r["lenient"]).astype(bool)
        s["lenient_perturbed"] = float(le[on].mean()) if on.any() else None
        s["lenient_unperturbed"] = float(le[~on].mean()) if (~on).any() else None
    elif env.tcfg.hold_cap_s != JE.T.TaskConfig.hold_cap_s or env.cfg.hold_obs_s:
        r["hold_cap"] = np.array(json.dumps(dict(hold_cap_s=env.tcfg.hold_cap_s, hold_obs_s=env.cfg.hold_obs_s)))
    r["summary"] = np.array(json.dumps(s))
    r["jt"] = np.array(json.dumps(jt.info()))
    r["world"] = np.array(a.world)
    if JE.EVAL_KP_SCALE != 1.0:   # $NEDM_SO101_EVAL_KP_SCALE (also when the robot dynamics are off)
        r["kp_scale"] = np.float64(JE.EVAL_KP_SCALE)
    if a.out:
        np.savez_compressed(a.out, **r)
    log(f"JT {a.world} ({B} tasks): {json.dumps(s)}")


def cmd_eval_expert(a):
    from nedm.so101_policy.vision import teacher as TT
    te = TT.Teacher(x(a.teacher))
    t = load_tasks(a.tasks, a.n, a.first)
    d = JT.expert_collect(t, te, a.world, 0.0, 0, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon,
                          len(t["seeds"]), log, record=False)
    s = JT.summarize(dict(success=d["success"], lenient=d["lenient"], end=d["end"],
                          success10=np.zeros_like(d["success"]), lifted=d["lifted"], end_xy=d["end_xy"]))
    if a.out:
        np.savez_compressed(a.out, **d, summary=np.array(json.dumps(s)))
    log(f"current teacher + decoder v3, {a.world}: {json.dumps(s)}")


def cmd_ppo(a):
    from nedm.so101_policy import ppo as P
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=1))
    init = torch.load(x(a.init), map_location=a.device, weights_only=False)
    codec = JE.Codec(np.asarray(init["env"]["codec"]["mean"]), np.asarray(init["env"]["codec"]["std"]))
    env_over = json.loads(Path(a.env_config).read_text()) if a.env_config else {}
    cfg = JE.JointEnvConfig.from_dict({**dict(world="nrd", num_envs=a.envs, device=a.device, seed=a.seed,
                                              ckpts=[x(a.nrd_ckpt)]), **env_over})
    env = JE.JointVecEnv(cfg, codec)
    if env.roll_hi is not None:   # the wrist-roll cap of the env config (roll_hi_deg): also in the dev evaluations
        log(f"roll cap: {json.dumps(env.roll_info())}")
    if env.rest is not None:      # held-cube rest starts of the env config (rest; round v9)
        log(f"rest starts: {json.dumps(env.rest_info())}")
    if env.pause is not None:     # the arm pause of the env config (pause)
        log(f"pause: {json.dumps(asdict(cfg.pause))}")
    if cfg.far_open is not None:  # the jaw-open cost far from the goal (far_open; round v9)
        log(f"far-open jaw cost: {json.dumps(asdict(cfg.far_open))}")
    tc = JT.ppo_train_cfg(json.loads(Path(a.train_config).read_text()) if a.train_config else None)
    tc["seed"] = a.seed
    pk = dict(init["policy_kwargs"])
    pk.update(init_std=tc["init_std"])
    pol = P.Policy(**pk).to(a.device)
    sd = init["policy"]
    own = pol.state_dict()
    for k in ("std",):
        sd = {**sd, k: own[k]}
    pol.load_state_dict(sd)
    ref = None
    if a.bc_coef > 0:
        import copy
        ref_pol = copy.deepcopy(pol).eval()
        for p_ in ref_pol.parameters():
            p_.requires_grad_(False)
        ref = lambda o: ref_pol.actor(o)                                           # noqa: E731 (normalised obs)
    algo = JT.JointPPO(env, pol, tc, ref, a.bc_coef, a.critic_warmup)
    start = 0
    if a.resume:
        d = algo.load(x(a.resume))
        start = algo.iteration
    curve = P.CurveWriter(out / "learning_curve.csv")
    dev_t = load_tasks(a.dev_tasks) if a.dev_tasks else None
    # more development sets (--dev-tasks-extra): the same evaluation, in the log and dev_evals.jsonl only
    dev_x = [(Path(p_).stem, load_tasks(p_)) for p_ in a.dev_tasks_extra or []]
    if dev_x and dev_t is None:
        raise SystemExit("--dev-tasks-extra needs --dev-tasks")
    dev_kw = {}                   # --dev-match-env: the training env's episode length, hold cap and held-time scale
    if a.dev_match_env:
        dev_kw = dict(horizon=int(cfg.horizon), hold_cap_s=env.tcfg.hold_cap_s, hold_obs_s=float(cfg.hold_obs_s))
        log(f"dev evaluations: {json.dumps(dev_kw)} (the training env's)")
    t0 = time.time()
    for it in range(start, a.iterations):
        row = algo.iterate()
        dev_more = []
        if dev_t is not None and (algo.iteration % a.eval_every == 0 or algo.iteration == a.iterations):
            jt_path = out / "ckpt" / f"model_{algo.iteration:05d}.pt"
            algo.save(jt_path)
            jt = JE.JointTeacher(jt_path, a.device)
            rec = dict(iteration=algo.iteration, sets={})
            for k_, (name, tt) in enumerate([(Path(a.dev_tasks).stem, dev_t)] + dev_x):
                ev = JT.make_env("nrd", len(tt["seeds"]), jt.codec, a.device, x(a.nrd_ckpt),
                                 roll_hi_deg=cfg.roll_hi_deg, **dev_kw)
                r = JT.run_tasks(ev, tt, jt)
                ev.close()
                s = JT.summarize(r)
                rec["sets"][name] = s
                if k_ == 0:       # the first set: the learning curve's dev columns (as before)
                    row["dev_nrd_lenient"], row["dev_nrd_strict"] = s["lenient"], s["strict"]
                    if "roll_cmd_over" in s:
                        row["dev_nrd_roll_cmd_over"] = s["roll_cmd_over"]
                else:
                    dev_more.append(f" | {name} len {s['lenient']:.2f} strict {s['strict']:.2f}"
                                    + (f" roll over {s['roll_cmd_over']:.2f} at cap {s['roll_q_at_cap']:.2f}"
                                       if "roll_cmd_over" in s else ""))
            with open(out / "dev_evals.jsonl", "a") as f_:   # every dev set's summary (the CSV keeps its header)
                f_.write(json.dumps(rec) + "\n")
        curve.write(row)
        log(f"it {algo.iteration}: succ {row['success']:.3f} len {row['lenient']:.3f} lifted {row['lifted']:.3f} "
            f"kl {row['kl']:.4f} lr {row['lr']:.2e} vloss {row['value_loss']:.4f} std {row['std_move']:.3f}/"
            f"{row['std_trigger']:.3f} {row['world_steps_per_s']:.0f} st/s"
            + (f" | roll over cap {row['roll_cmd_over']:.3f}" if "roll_cmd_over" in row else "")
            + (f" | dev nrd len {row['dev_nrd_lenient']:.2f} strict {row['dev_nrd_strict']:.2f}"
               if "dev_nrd_lenient" in row else "")
            + (f" roll over {row['dev_nrd_roll_cmd_over']:.2f}" if "dev_nrd_roll_cmd_over" in row else "")
            + "".join(dev_more)
            + (f" | rest starts {env.rest_n}/{env.rest_resets}" if env.rest is not None else "")
            + (f" | pauses {env.pause.started}" if env.pause is not None else "")
            + (f" | far-open cost {float(env.far_open_sum):.1f}" if cfg.far_open is not None else ""))
        if algo.iteration % a.save_every == 0 or algo.iteration == a.iterations:
            algo.save(out / "ckpt" / f"model_{algo.iteration:05d}.pt")
            algo.save(out / "ckpt" / "last.pt")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--device", default="cuda")
        q.add_argument("--nrd-ckpt", default=NRD)
        q.add_argument("--teacher", default=TEACHER)
        q.add_argument("--workers", type=int, default=8)
        q.add_argument("--mj-python", default=MJ_PY)
        q.add_argument("--horizon", type=int, default=400)
        q.add_argument("--seed", type=int, default=0)

    q = sub.add_parser("tasks")
    q.add_argument("--kind", choices=("C", "bank", "H"), default="C")
    q.add_argument("--n", type=int, required=True)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--dev", action="store_true")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--home-config", default="", help="kind H: JSON with home_deg, noise_deg, bank")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_tasks)

    q = sub.add_parser("mix")
    q.add_argument("--inputs", nargs="+", required=True)
    q.add_argument("--counts", type=int, nargs="+", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_mix)

    q = sub.add_parser("starts")
    q.add_argument("--inputs", nargs="+", required=True, help="start banks (.npz) and training task files")
    q.add_argument("--repeat", type=int, nargs="+", default=[1], help="times every row of input i (one or per input)")
    q.add_argument("--exclude", nargs="*", default=[], help="task files (dev / fresh) no task-file seed may be in")
    q.add_argument("--roll-hi-deg", type=float, default=None, help="drop rows with the wrist roll above (sim deg)")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_starts)

    q = sub.add_parser("expert")
    common(q)
    q.add_argument("--world", default="nrd")
    q.add_argument("--tasks", default="")
    q.add_argument("--n", type=int, default=1024)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--noise-frac", type=float, default=0.5)
    q.add_argument("--noise", type=float, default=1.0)
    q.add_argument("--batch", type=int, default=1024)
    q.add_argument("--no-record", action="store_true")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_expert)

    q = sub.add_parser("bc")
    common(q)
    q.add_argument("--data", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    q.add_argument("--steps", type=int, default=20000)
    q.add_argument("--batch", type=int, default=4096)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--rate-noise", type=float, default=0.0)
    q.set_defaults(fn=cmd_bc)

    q = sub.add_parser("eval")
    common(q)
    q.add_argument("--jt", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--world", default="mujoco")
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--noise", type=float, default=0.0)
    q.add_argument("--no-lead", action="store_true")
    q.add_argument("--no-floor", action="store_true")
    q.add_argument("--record-states", action="store_true")
    q.add_argument("--hold-cap-s", type=float, default=None, help="the hold_cap rule (s held in the air; default 3.5)")
    q.add_argument("--hold-obs-s", type=float, default=0.0,
                   help="held-time scale of the JT observation (s; 0: the task's, 3.5 with another --hold-cap-s)")
    q.add_argument("--roll-hi-deg", type=float, default=None,
                   help="the wrist-roll target cap (sim deg; the real roll ends at 101.2); default off")
    q.add_argument("--out", default="")
    from nedm.so101_policy import table as TB
    TB.add_cli(q)
    q.set_defaults(fn=cmd_eval)

    q = sub.add_parser("eval-expert")
    common(q)
    q.add_argument("--tasks", required=True)
    q.add_argument("--world", default="mujoco")
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--out", default="")
    q.set_defaults(fn=cmd_eval_expert)

    q = sub.add_parser("ppo")
    common(q)
    q.add_argument("--init", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--envs", type=int, default=4096)
    q.add_argument("--iterations", type=int, default=200)
    q.add_argument("--env-config", default="")
    q.add_argument("--train-config", default="")
    q.add_argument("--bc-coef", type=float, default=0.0)
    q.add_argument("--critic-warmup", type=int, default=10)
    q.add_argument("--dev-tasks", default="")
    q.add_argument("--dev-tasks-extra", nargs="+", default=None,
                   help="more development sets (same evaluation; the log and dev_evals.jsonl, not the CSV)")
    q.add_argument("--dev-match-env", action="store_true",
                   help="dev evaluations with the env config's horizon, hold cap and held-time scale (default: 400 "
                        "steps and the task defaults)")
    q.add_argument("--eval-every", type=int, default=10)
    q.add_argument("--save-every", type=int, default=10)
    q.add_argument("--resume", default="")
    q.set_defaults(fn=cmd_ppo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
