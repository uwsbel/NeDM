"""VIRAL / DoorMan-style camera student of build9 (``nedm.so101_policy.vision.viral``; docs
~/so101_policy/docs/build9_viral_student.md). The teacher is the Markov joint teacher (``vision_viral_teacher.py``).

    export PYTHONPATH=src PY=~/miniconda3/envs/so101_vision/bin/python
    $PY -P scripts/so101_policy/vision_viral.py run --out RUN --jt JT.pt --tasks train.npz --eval-tasks dev100.npz \
        --config configs/so101_policy/vision/viral_v1.json --r0-tasks 2048 --rounds 4 --round-tasks 1024 \
        --teacher-share 0.5
    $PY -P scripts/so101_policy/vision_viral.py eval --student RUN/r4/last.pt --jt JT.pt --tasks dev100.npz \
        --world mujoco [--randomise] [--record-states] --out e.npz
    $PY -P scripts/so101_policy/vision_viral.py collect --jt JT.pt --tasks t.npz --world nrd --out d.npz [--student S]

``run`` (VIRAL's mixing per WHOLE episode, never inside one): round 0 = JT-driven episodes (``--r0-noise-frac`` of them
with noise ``--noise`` x the JT's PPO std on its normalised action); round i >= 1 = ``--round-tasks`` new tasks, the
share ``--teacher-share`` JT-driven (the same noise mix), the rest driven by the previous round's student with
randomised visuals (``--collect-randomise``); every visited decision labelled by the JT. After each round: train
(round 0 from ImageNet, later rounds fine-tune on the aggregate), evaluate on the development tasks in CPU MuJoCo (and
``--eval-worlds``), nominal visuals (+ randomised with ``--eval-randomise``). Files resume.

Real2sim (``~/so101_r2s_claude/docs/integrate.md``): a config with ``scene`` renders the real scene (``viral.make_views``;
the scene / calibration sha256 are pinned in the student config at the start of ``run`` and checked later:
``--allow-scene-change``); a config with image lag (``lag``, ``lag_range``) records the 20 ms states in the round files
(``row_lag``) and renders the lagged state everywhere (training, the online loop, the evaluations; ``eval --lag K``
overrides the nominal lag). ``--eval-tasks`` takes several files (the first one keeps the old output names; the others
add their file stem).

Robot timing and servo (``joint_env.RobotDynConfig``, all off by default): training = the online config's ``dyn`` (per
world per episode; with ``qd_robot_p``, ``horizon``, ``hold_cap_s``, ``qd_noise_zero_p``, ``lag_range``,
``viral_r2s_online_dyn_v1.json``); evaluation / collection = one fixed setting for the process in
``NEDM_SO101_DYN='{"d": 2, "jitter": 0.4, "s": 0.5, "tau": 0.1, "sag_deg": 1.5, "i_time_s": 0.2}'`` (recorded as
"dyn" in eval files, "robot" + ``dyn_params`` in round files), with ``--horizon 1500 --hold-cap-s 20`` for slow worlds
and ``eval --qd-robot 3`` for the robot's qdot estimate (``row_qd_input``: the student's qdot input). jt_h110 scores
0 % in that robot-like setting (NRD and MuJoCo); settings with a teacher reference: the config note.

Last-command input (``ViralConfig.lastcmd``, off by default): ``online --config viral_r2s_v1_lastcmd.json`` adds it to
a student without it (zero weights: the initial student's actions at the start); ``eval`` / ``collect`` feed it from
the env's executed targets; ``run`` (round files) refuses it.

Studentlab (2026-10-22, ~/so101_studentlab_claude/trials/BUILD.md): ``online`` expands the loaded student when the
config adds inputs (``hist``, ``cmd_in``, ``img_hist``: ``viral.expand_student``, zero-init) and takes
``goal_jump`` / ``hold_cap_s`` in the online config (arms: ``configs/so101_policy/vision/studentlab/``,
``cluster/studentlab_online.sbatch``); ``eval --goal-shift 'p=1,d=10:25,lat=8'`` is the evaluation-only goal shift
(the overshoot gate: once per episode, same trigger as the training jumps, the cube ends PAST the moved goal; the
summary adds ``goal_shift`` with the shifted / unshifted episodes). The online config's ``light_redraw_p`` (< 1: the
per-call light and intrinsics stay for several decisions; the image-history arm A4) also applies to the in-run
evaluations; ``eval --light-redraw-p P`` is the same for a standalone evaluation (with ``--randomise``).

Table randomisation (real2sim v8, off by default; ``nedm.so101_policy.table_bias``): the online config's
``table_bias`` puts a per-world table plane under the student's joint reading (q, qdot, cmd inputs, labels and applied
targets in that frame), ``table_render`` draws the training images consistently (the arm at the reading, the world
on the plane), ``plate_mm`` randomises the goal plate thickness of the TRAINING views only
(``viral.online_views_cfg``; the evaluation views and the saved student config keep the scene's plate); configs
``viral_r2s_online_v8_C_table.json`` / ``viral_r2s_online_v8_S_table.json``. The gate (pass rule in its docstring) is
``scripts/so101_policy/real2sim/lowtable_gate.py`` (``eval --table-plane`` / ``--floor table``: the physical table
moved in CPU MuJoCo); robot deploy with ``run_student_real.py --table-plane calib``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import joint_teacher as JT
from nedm.so101_policy.vision import viral as VI

NRD = "~/so101_policy/ckpt/nrd_c1_s62.pt"
MJ_PY = "~/miniconda3/envs/nrdsim/bin/python"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def x(p):
    return str(Path(p).expanduser()) if p else p


def progress(out, **kw):
    with open(Path(out) / "progress.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw), default=float) + "\n")


def load_tasks(path, idx=None):
    from nedm.so101_policy.vision.demos import load_tasks as lt
    return lt(path, idx)


def subset(t, idx):
    n = len(t["seeds"])
    return {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in t.items()}


class EnvCache:
    def __init__(self, jt, device, nrd, workers, mj_python, horizon, table=None, cli=None, hold_cap_s=None,
                 goal_jump=None):
        self.c, self.jt, self.dev, self.nrd, self.w, self.mp, self.h = {}, jt, device, nrd, workers, mj_python, horizon
        self.table = table        # (plane, floor, q_offset) for JointVecEnv.set_table (evaluation only), None: off
        self.cli = cli            # eval args (``--log-only-extra``), None: off
        self.hold_cap_s = hold_cap_s   # the hold_cap rule (s; ``--hold-cap-s``), None: the task default
        self.goal_jump = goal_jump     # JE.GoalJumpConfig: the evaluation-only goal shift (``--goal-shift``), None: off

    def get(self, world, B):
        k = (world, B)
        if k not in self.c:
            self.c[k] = JT.make_env(world, B, self.jt.codec, self.dev, self.nrd, self.w, self.mp, self.h,
                                    hold_cap_s=self.hold_cap_s, goal_jump=self.goal_jump)
            if self.c[k].goal_jump is not None:
                log(f"goal shift ({world}, {B} worlds): {json.dumps(VI.asdict(self.c[k].goal_jump.cfg))}")
            if self.c[k].dyn is not None:      # $NEDM_SO101_DYN (joint_teacher.make_env)
                log(f"robot dynamics ({world}, {B} worlds): {json.dumps(VI.asdict(self.c[k].dyn.cfg))}")
            if self.table is not None:
                log(f"table: {json.dumps(self.c[k].set_table(*self.table))}")
            if self.cli is not None:
                from nedm.so101_policy import table as TB
                if TB.log_only_extra(self.c[k], self.cli):
                    log(f"log-only task rules: {self.c[k].tcfg.log_only}")
        return self.c[k]


def taps(env, *fns):
    """``viral.step_tap`` with the given 20 ms state callbacks (none: a no-op context)."""
    fns = [f for f in fns if f is not None]
    return VI.step_tap(env, *fns) if fns else contextlib.nullcontext()


def robot_rec(env):
    """The env's robot setting for output files (JSON string): the robot dynamics config (``RobotDynConfig``; the
    per-world draws are separate), the hold cap / held-time scale and ``$NEDM_SO101_EVAL_KP_SCALE``; None = all
    defaults (the old env)."""
    if env.dyn is None and env.tcfg.hold_cap_s == JE.T.TaskConfig.hold_cap_s and not env.cfg.hold_obs_s \
            and JE.EVAL_KP_SCALE == 1.0:
        return None
    return json.dumps(dict(cfg=VI.asdict(env.dyn.cfg) if env.dyn is not None else None, cols=list(JE.RobotDyn.COLS),
                           hold_cap_s=env.tcfg.hold_cap_s, hold_obs_s=float(env.cfg.hold_obs_s),
                           kp_scale=JE.EVAL_KP_SCALE))


@torch.no_grad()
def collect(tasks, world, jt, cache: EnvCache, src, noise, seed, batch, student=None, log=log, lag_rec=False,
            lag_kmax=0):
    """Episodes on all tasks in batches -> concatenated record dict (``run_tasks`` layout); ``lag_rec``: also the
    states of the last max(``viral.LAG_REC``, ``lag_kmax``) control steps before every row (``row_lag``, image
    lag). Robot dynamics (``$NEDM_SO101_DYN``): per-episode draws ``dyn_params`` (``RobotDyn.COLS``) and the setting
    "robot" (``robot_rec``)."""
    n = len(tasks["seeds"])
    parts, t0, rec = [], time.time(), None
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        env = cache.get(world, len(idx))
        if student is not None:
            student.bind(env)             # cmd_in and the last-command input (ViralConfig.lastcmd): the env's u_exe
        sl = VI.SubstepLog() if lag_rec else None
        push = student.push if student is not None and student.needs_tap else None
        with taps(env, sl, push):
            r = JT.run_tasks(env, subset(tasks, idx), jt, student, src[idx], noise[idx], seed, record=True)
        if env.dyn is not None:
            r["dyn_params"] = env.dyn.params().cpu().numpy().astype(np.float32)
        rec = robot_rec(env)
        if sl is not None:
            r["row_lag"] = VI.lag_rows(r, sl.array(), max(VI.LAG_REC, int(lag_kmax)))
        parts.append(r)
        s = np.asarray(r["src"]) == 1
        msg = f"  collect {world}: tasks {s0}-{idx[-1]}, lenient {r['lenient'].mean() * 100:.1f} %"
        if s.any():
            msg += f" (student-driven {r['lenient'][s].mean() * 100:.1f} % of {int(s.sum())}"
            msg += f", JT-driven {r['lenient'][~s].mean() * 100:.1f} %)" if (~s).any() else ")"
        log(msg + f", {time.time() - t0:.0f} s")
    out = concat(parts)
    if rec is not None:           # after concat (it drops 0-d entries)
        out["robot"] = np.array(rec)
    return out


def concat(parts):
    if len(parts) == 1:
        return parts[0]
    out, e0, r0 = {}, 0, 0
    for p in parts:
        q = dict(p)
        q["row_ep"] = p["row_ep"].astype(np.int64) + e0
        q["ep_row_start"] = p["ep_row_start"] + r0
        e0 += len(p["end"])
        r0 += len(p["row_ep"])
        for k, v in q.items():
            if np.ndim(v) == 0:
                continue
            out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["wall_s"] = np.float64(sum(float(p["wall_s"]) for p in parts))
    return out


def evaluate(st, jt, tasks, world, cache, views, randomise=False, jaw_lead=None, record_states=False,
             full_horizon=False, device="cuda", lag=None, lag_range=None, qd_window=0, light_redraw_p=1.0):
    """``lag``: image lag in control steps (default the student's nominal ``cfg.lag``); ``lag_range``: drawn per world
    instead; ``qd_window`` > 0: the robot-like qdot input (``viral.robot_qd``); ``light_redraw_p`` < 1: the per-call
    light / intrinsics stay for several decisions (``viral.Actor``)."""
    actor = VI.Actor(st, views, randomise=randomise, jaw_lead=jaw_lead, device=device, lag=lag, lag_range=lag_range,
                     qd_window=qd_window, light_redraw_p=light_redraw_p)
    env = cache.get(world, len(tasks["seeds"]))
    actor.bind(env)                                     # cmd_in and the last-command input (ViralConfig.lastcmd):
                                                        # the env's executed targets (env.u_exe)
    if getattr(env, "q_offset", None) is not None:      # evaluation only (table.py): the student's joint frame
        actor.q_offset = env.q_offset.tolist()
    with taps(env, actor.push if actor.needs_tap else None):
        r = JT.run_tasks(env, tasks, jt, actor, np.ones(len(tasks["seeds"]), np.int64), None, 0,
                         record=True, record_states=record_states, full_horizon=full_horizon)
    r["render_s"], r["model_s"] = np.float64(actor.t_render), np.float64(actor.t_model)
    if getattr(env, "table_plane", None) is not None or getattr(env, "q_offset", None) is not None \
            or getattr(env, "floor_z", 0.0) != 0.0 or getattr(env, "floor_plane", None) is not None:
        r["table"] = np.array(json.dumps(env.table_info()))
    if cache.cli is not None:
        r["log_only"] = np.array(json.dumps(list(env.tcfg.log_only)))
    summ = JT.summarize(r)
    if env.goal_jump is not None:     # the evaluation-only goal shift: outcomes of the shifted / unshifted episodes
        summ["goal_shift"] = goal_shift_summary(r)
    r["summary"] = np.array(json.dumps(summ))
    r["world"] = np.array(world)
    r["randomise"] = np.bool_(randomise)
    r["jaw_lead"] = np.int64(actor.jaw_lead)
    if actor.light_redraw_p < 1.0:  # per-call light / intrinsics kept for several decisions: the setting, the changes
        r["light_redraw"] = np.array(json.dumps(dict(p=actor.light_redraw_p, changes=actor.state_changes,
                                                     decisions=actor.calls)))
    if actor.kmax > 0:
        r["image_lag"] = np.array(json.dumps(dict(lag=actor.lag, lag_range=actor.lag_range)))
    if actor.qd_window > 0:       # the robot-like qdot: window and the student's qdot INPUT per row (row_qd: the sim's)
        r["qd_robot"] = np.int64(actor.qd_window)
        if actor.qd_log and "row_ep" in r:
            r["row_qd_input"] = torch.stack(actor.qd_log, 1).numpy()[r["row_ep"].astype(np.int64),
                                                                       r["row_t"].astype(np.int64)]
    if env.dyn is not None:       # robot dynamics of the evaluation ($NEDM_SO101_DYN): the setting + per-episode draws
        r["dyn"] = np.array(json.dumps(env.dyn_info()))
    elif env.tcfg.hold_cap_s != JE.T.TaskConfig.hold_cap_s:
        r["hold_cap_s"] = np.float64(env.tcfg.hold_cap_s)
    if JE.EVAL_KP_SCALE != 1.0:   # $NEDM_SO101_EVAL_KP_SCALE (also when the robot dynamics are off)
        r["kp_scale"] = np.float64(JE.EVAL_KP_SCALE)
    if st.cfg.scene:
        r["scene"] = np.array(json.dumps(dict(scene=st.cfg.scene, scene_sha256=st.cfg.scene_sha256,
                                              calib_sha256=st.cfg.calib_sha256, crops=st.cfg.crops)))
    return r


def goal_shift_summary(r):
    """Outcomes of the episodes with / without a goal jump (``run_tasks`` output with ``goal_jumps``): count, lenient,
    strict, end xy error p50 / p90 (mm; to the final goal), failure causes; the shift offsets (mm, + = past)."""
    from nedm.so101_policy import task as T
    n = np.asarray(r["goal_jumps"]) > 0
    le, s = np.asarray(r["lenient"]).astype(bool), np.asarray(r["success"]).astype(bool)
    ex = np.asarray(r["end_xy"], np.float64) * 1e3
    out = {}
    for tag, m in (("shifted", n), ("unshifted", ~n)):
        if not m.any():
            out[tag] = dict(n=0)
            continue
        e = ex[m][np.isfinite(ex[m])]
        causes = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[m & ~le], return_counts=True))}
        out[tag] = dict(n=int(m.sum()), lenient=float(le[m].mean()), strict=float(s[m].mean()),
                        end_xy_mm_p50=float(np.median(e)) if len(e) else None,
                        end_xy_mm_p90=float(np.percentile(e, 90)) if len(e) else None, causes_not_lenient=causes)
    if n.any():
        d = np.asarray(r["goal_jump_d"], np.float64)[n] * 1e3
        out["shift_d_mm"] = [float(v) for v in np.percentile(d, [0, 50, 100])]
    return out


def eval_sets(paths, n=0):
    """``--eval-tasks`` files -> [(tag prefix, tasks)]: the first file keeps the old names (prefix ""), the others
    are prefixed with their file stem."""
    out = []
    for i, p_ in enumerate(paths):
        out.append(("" if i == 0 else Path(p_).stem + "_", load_tasks(p_, np.arange(n) if n else None)))
    return out


def guard(tasks, paths):
    """No training task may be an evaluation task."""
    tr = set(np.asarray(tasks["seeds"]).tolist())
    for p_ in paths:
        hit = tr & set(np.asarray(load_tasks(p_)["seeds"]).tolist())
        if hit:
            raise ValueError(f"{len(hit)} training task seeds are in {p_}")


def agree(r_a, r_b):
    a, b = np.asarray(r_a["lenient"]).astype(bool), np.asarray(r_b["lenient"]).astype(bool)
    return dict(both=int((a & b).sum()), a_only=int((a & ~b).sum()), b_only=int((~a & b).sum()),
                neither=int((~a & ~b).sum()), agreement=float((a == b).mean()))


def cmd_run(a):
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if not (out / "args.json").exists():
        (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=1, default=str))
    jt = JE.JointTeacher(x(a.jt), a.device)
    cfg_d = json.loads(Path(a.config).read_text())
    for s in a.set or []:
        k, v = s.split("=", 1)
        cfg_d[k] = json.loads(v)
    cfg_d["seed"] = a.seed
    cfg = VI.ViralConfig.from_dict(cfg_d)
    VI.pin_real(cfg, a.allow_scene_change)              # real scene: crops + file hashes into the student config
    if cfg.scene:                                        # a resumed run must keep its scene (rounds share one scene)
        pin = dict(scene=cfg.scene, scene_sha256=cfg.scene_sha256, calib_sha256=cfg.calib_sha256, crops=cfg.crops)
        pf = out / "real_pin.json"
        if pf.exists():
            old = json.loads(pf.read_text())
            diff = [k for k in ("scene_sha256", "calib_sha256", "crops") if old.get(k) != pin[k]]
            if diff and not a.allow_scene_change:
                raise ValueError(f"{pf}: {diff} changed since the run started (--allow-scene-change to continue)")
        else:
            pf.write_text(json.dumps(pin, indent=1))
    tasks = load_tasks(a.tasks)
    from nedm.so101_policy.vision.teacher import check_seeds
    check_seeds(tasks["seeds"])
    evs = eval_sets(a.eval_tasks, a.eval_n)
    guard(tasks, list(a.eval_tasks) + list(a.guard_tasks or []))
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.eval_workers, x(a.mj_python), a.horizon, hold_cap_s=a.hold_cap_s)
    views = VI.make_views(cfg, a.render_worlds, seed=a.seed, allow_change=a.allow_scene_change, device=a.device)
    lag_rec = cfg.lag_max > 0
    rows = [np.arange(0, a.r0_tasks)]
    for i in range(a.rounds):
        s0 = a.r0_tasks + i * a.round_tasks
        rows.append(np.arange(s0, s0 + a.round_tasks))
    if rows[-1][-1] >= len(tasks["seeds"]):
        raise ValueError(f"{len(tasks['seeds'])} tasks, need {rows[-1][-1] + 1}")
    files = [x(f) for f in (a.extra_data or [])]
    prev = x(a.init_from) if a.init_from else None
    for i in range(a.rounds + 1):
        f = out / "data" / f"r{i}.npz"
        t = subset(tasks, rows[i])
        n = len(rows[i])
        if not f.exists():
            t0 = time.time()
            rng = np.random.default_rng(a.seed * 1000 + i)
            share = 1.0 if i == 0 else a.teacher_share
            src = np.where(rng.random(n) < share, 0, 1).astype(np.int64)
            if prev is None:
                src[:] = 0
            noisy = (src == 0) & (rng.random(n) < a.r0_noise_frac)
            noise = np.where(noisy, a.noise, 0.0)
            student = None
            if (src == 1).any():
                st, _ = VI.load_student(prev, a.device)
                student = VI.Actor(st, views, randomise=a.collect_randomise, device=a.device,
                                   lag_range=cfg.lag_range, seed=a.seed + i)
            d = collect(t, a.world, jt, cache, src, noise, a.seed + 1000 * i, a.batch, student, lag_rec=lag_rec,
                        lag_kmax=cfg.lag_max)
            d["meta"] = np.array(json.dumps(dict(round=i, world=a.world, jt=jt.info(), teacher_share=share,
                                                 noise=a.noise, student=str(prev) if student else "",
                                                 collect_randomise=a.collect_randomise, wall_s=time.time() - t0)))
            np.savez_compressed(f, **d)
            s = d["src"] == 1
            progress(out, stage=f"data r{i}", wall_s=time.time() - t0, rows=int(len(d["row_ep"])),
                     lenient_jt=float(d["lenient"][~s].mean()) if (~s).any() else None,
                     lenient_student=float(d["lenient"][s].mean()) if s.any() else None, n_student=int(s.sum()))
            del d, student
            torch.cuda.empty_cache()
        files.append(str(f))
        ck = out / f"r{i}" / "last.pt"
        if not ck.exists():
            t0 = time.time()
            data = VI.StudentSet(files, a.device, cfg.chunk, seed=a.seed, lag=cfg.lag_max)
            first = i == 0 and not a.init_from
            steps = a.train_steps0 if first else a.train_steps
            VI.train(data, views, cfg, out / f"r{i}", max_steps=steps, init_from=None if first else prev, log=log,
                     device=a.device, extra_meta=dict(argv=sys.argv))
            progress(out, stage=f"train r{i}", wall_s=time.time() - t0, steps=steps, rows=data.n_rows,
                     eps=data.n_eps)
            del data
            torch.cuda.empty_cache()
        prev = str(ck)
        if i % a.eval_every == 0 or i == a.rounds:
            st, _ = VI.load_student(ck, a.device)
            res = {}
            for pre, ev, w, rnd in [(p_, e_, w_, r_) for p_, e_ in evs for w_ in a.eval_worlds
                                    for r_ in ([False, True] if a.eval_randomise else [False])]:
                tag = f"{pre}{w}{'_dr' if rnd else ''}"
                fe = out / f"r{i}" / f"eval_{tag}.npz"
                if not fe.exists():
                    t0 = time.time()
                    r = evaluate(st, jt, ev, w, cache, views, rnd, device=a.device)
                    np.savez_compressed(fe, **r)
                    s = json.loads(str(r["summary"]))
                    progress(out, stage=f"eval r{i} {tag}", wall_s=time.time() - t0, **s)
                    log(f"r{i} eval {tag}: lenient {s['lenient'] * 100:.0f} % strict {s['strict'] * 100:.0f} % "
                        f"lifted {s['lifted'] * 100:.0f} % causes {s['causes_not_lenient']}")
                res[tag] = dict(np.load(fe))
            if "mujoco" in res and "nrd" in res:
                progress(out, stage=f"agree r{i}", **agree(res["mujoco"], res["nrd"]))
            del st


def cmd_online(a):
    """VIRAL's online loop (``viral.OnlineDAgger``) from an initial student checkpoint; config overrides of the student
    (``--config`` keys that differ, e.g. qd_noise, jaw_lead, dr) are applied to the loaded student."""
    import csv
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=1, default=str))
    jt = JE.JointTeacher(x(a.jt), a.device)
    st, ck = VI.load_student(x(a.init), a.device)
    if a.config:
        over = json.loads(Path(a.config).read_text())
        for s_ in a.set or []:
            k, v = s_.split("=", 1)
            over[k] = json.loads(v)
        from dataclasses import asdict as _ad
        cd = _ad(st.cfg)
        arch = {k: over.get(k, cd[k]) for k in VI.ARCH_KEYS}       # input options: zero-init expansion of the student
        lastcmd = bool(over.get("lastcmd", cd["lastcmd"]))           # the last-command input (own zero-init layer)
        if "cmd_scale" in over and st.cfg.cmd_in and tuple(over["cmd_scale"]) != tuple(st.cfg.cmd_scale):
            raise SystemExit("cmd_scale: the student already has a cmd input with another scale")
        cd.update({k: v for k, v in over.items() if k in cd and k not in VI.ARCH_KEYS and k != "lastcmd"})
        st.cfg = VI.ViralConfig.from_dict(cd)
        if (int(arch["hist"]), bool(arch["cmd_in"]), bool(arch["img_hist"])) != \
                (int(st.cfg.hist), bool(st.cfg.cmd_in), bool(st.cfg.img_hist)):
            st = VI.expand_student(st, hist=int(arch["hist"]), cmd_in=bool(arch["cmd_in"]),
                                   img_hist=bool(arch["img_hist"]), device=a.device)
            log(f"student inputs: hist {st.cfg.hist}, cmd_in {st.cfg.cmd_in}, img_hist {st.cfg.img_hist} "
                f"(zero-init expansion of {a.init})")
        if lastcmd != st.cfg.lastcmd:                    # + the last-command input, zero weights (acts as the init)
            if not lastcmd:
                raise SystemExit("--config lastcmd false: the initial student has the last-command input")
            st = VI.expand_lastcmd(st, a.device)
            log("student: + the last-command input (zero-initialised; ViralConfig.lastcmd)")
    VI.pin_real(st.cfg, a.allow_scene_change)
    oc = json.loads(Path(a.online_config).read_text()) if a.online_config else {}
    for s_ in a.oset or []:
        k, v = s_.split("=", 1)
        oc[k] = json.loads(v)
    ocfg = VI.OnlineConfig.from_dict(oc)
    if ocfg.lag_range and tuple(ocfg.lag_range) != tuple(st.cfg.lag_range):      # the online config's image lag range
        sets = dict(s_.split("=", 1) for s_ in a.set or [])                       # (wins over --config, not --set)
        if a.config and "lag_range" in sets and tuple(json.loads(sets["lag_range"])) != tuple(ocfg.lag_range):
            raise SystemExit(f"lag_range: --set {sets['lag_range']} vs the online config {list(ocfg.lag_range)}")
        log(f"image lag range {list(ocfg.lag_range)} (online config; student {list(st.cfg.lag_range)})")
        st.cfg = VI.ViralConfig.from_dict({**VI.asdict(st.cfg), "lag_range": list(ocfg.lag_range)})
    tasks = load_tasks(a.tasks)
    from nedm.so101_policy.vision.teacher import check_seeds
    check_seeds(tasks["seeds"])
    evs = eval_sets(a.eval_tasks, a.eval_n)
    guard(tasks, list(a.eval_tasks) + list(a.guard_tasks or []))
    vcfg = VI.online_views_cfg(st.cfg, ocfg)       # training views: + the online config's plate thickness (v8)
    if vcfg is not st.cfg:
        log(f"training views: RealDR plate_thickness {vcfg.real_dr['plate_thickness']} m (online config plate_mm; "
            "the evaluation views and the student's config keep the scene's plate)")
    views = VI.make_views(vcfg, ocfg.envs, seed=a.seed, allow_change=a.allow_scene_change, device=a.device)
    eviews = VI.make_views(st.cfg, a.render_worlds, seed=a.seed + 1, allow_change=a.allow_scene_change, device=a.device)
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.eval_workers, x(a.mj_python), a.horizon, hold_cap_s=a.hold_cap_s)
    od = VI.OnlineDAgger(st, jt, tasks, x(a.nrd_ckpt), views, ocfg, a.device, a.seed, log)
    meta = dict(student_cfg=VI.cfg_dict(st.cfg), online=VI.asdict(ocfg), init=str(a.init), jt=jt.info(), argv=sys.argv)
    robot_opts = od.env.dyn is not None or od.qd_hist is not None or od.env.tcfg.horizon != 400 \
        or od.env.tcfg.hold_cap_s != JE.T.TaskConfig.hold_cap_s or od.env.goal_jump is not None
    if robot_opts:              # the resolved online env (robot dynamics, episode length, hold cap)
        meta["online_env"] = dict(horizon=od.env.tcfg.horizon, hold_cap_s=od.env.tcfg.hold_cap_s,
                                  hold_obs_s=od.env.cfg.hold_obs_s,
                                  dyn=VI.asdict(od.env.dyn.cfg) if od.env.dyn is not None else None)
        if od.env.goal_jump is not None:
            meta["online_env"]["goal_jump"] = VI.asdict(od.env.goal_jump.cfg)
        log(f"online env: {json.dumps(meta['online_env'])}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "curve.csv", "a", newline="")
    wr = None
    t0 = time.time()
    acc, tacc, n_acc = {}, {}, 0
    min_fill = max(st.cfg.batch * 4, ocfg.envs)
    base_lr = [g_["lr"] for g_ in od.opt.param_groups]
    for it in range(1, ocfg.iterations + 1):
        od.it = it - 1
        if ocfg.lr_decay_start >= 0 and it > ocfg.lr_decay_start:
            u = (it - ocfg.lr_decay_start) / max(1, ocfg.iterations - ocfg.lr_decay_start)
            for g_, b_ in zip(od.opt.param_groups, base_lr):
                g_["lr"] = b_ * (1 - (1 - ocfg.lr_final_frac) * u)
        T = od.act()
        if it == 1 and robot_opts:
            log(f"robot draws at the start: {json.dumps(od.log_extra())}")
        tt = time.time()
        info = od.train_steps() if od.buf.n >= min_fill else {}
        T["train"] = time.time() - tt
        for k, v in info.items():
            acc[k] = acc.get(k, 0.0) + v
        for k, v in T.items():
            tacc[k] = tacc.get(k, 0.0) + v
        n_acc += 1
        if it % ocfg.log_every == 0:
            s_ = od.stats
            row = dict(it=it, wall_min=round((time.time() - t0) / 60, 2), samples=od.samples, buffer=od.buf.n,
                       **{k: round(v / n_acc, 4) for k, v in acc.items()},
                       **{f"t_{k}": round(v / n_acc, 3) for k, v in tacc.items()},
                       ep_student=s_["ep_student"], len_student=round(s_["len_student"] / max(s_["ep_student"], 1), 3),
                       ep_teacher=s_["ep_teacher"], len_teacher=round(s_["len_teacher"] / max(s_["ep_teacher"], 1), 3),
                       **{k: s_[k] for k in ("ep_jump_student", "len_jump_student", "ep_jump_teacher",
                                             "len_jump_teacher") if k in s_},
                       **od.log_extra())
            od.stats = {k: 0 for k in s_}
            acc, tacc, n_acc = {}, {}, 0
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            log(json.dumps(row))
        if (ocfg.save_every > 0 and it % ocfg.save_every == 0) or it == ocfg.iterations:
            VI.save_student(st, out / "ckpt" / f"it_{it:05d}.pt", it=it, samples=od.samples)
            VI.save_student(st, out / "last.pt", it=it, samples=od.samples)
        if ocfg.eval_every > 0 and (it % ocfg.eval_every == 0 or it == ocfg.iterations):   # eval_every 0: none
            st.eval()
            for (pre, ev), rnd in [(e_, r_) for e_ in evs for r_ in ([False, True] if a.eval_randomise else [False])]:
                tag = f"{pre}mujoco{'_dr' if rnd else ''}"
                te = time.time()
                r = evaluate(st, jt, ev, "mujoco", cache, eviews, rnd, device=a.device,
                             light_redraw_p=ocfg.light_redraw_p)
                np.savez_compressed(out / f"eval_it{it:05d}_{tag}.npz", **r)
                s_ = json.loads(str(r["summary"]))
                progress(out, stage=f"eval it {it} {tag}", wall_s=time.time() - te, samples=od.samples, **s_)
                log(f"it {it} eval {tag}: lenient {s_['lenient'] * 100:.0f} % strict {s_['strict'] * 100:.0f} % "
                    f"lifted {s_['lifted'] * 100:.0f} % causes {s_['causes_not_lenient']}")
            st.train()
    if torch.cuda.is_available() and torch.device(a.device).type == "cuda":
        log(f"done: {ocfg.iterations} decisions in {(time.time() - t0) / 60:.1f} min, peak GPU memory (torch) "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def cmd_eval(a):
    from nedm.so101_policy import table as TB
    jt = JE.JointTeacher(x(a.jt), a.device)
    st, _ = VI.load_student(a.student, a.device)
    if a.scene_path:     # the same scene / calibration files at another path (the pinned sha256 is still checked)
        st.cfg.scene = x(a.scene_path)
    if a.calib_path:
        st.cfg.calib = x(a.calib_path)
    t = load_tasks(a.tasks, np.arange(a.first, a.first + a.n) if a.n else None)
    plane, floor, qo, moved = TB.from_cli(a)
    gs = JE.GoalJumpConfig.from_spec(a.goal_shift, max_n=1, before_p=0.0) if a.goal_shift else None
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon,
                     table=(plane, floor, qo) if moved else None, cli=a if a.log_only_extra else None,
                     hold_cap_s=a.hold_cap_s, goal_jump=gs)
    geo = {"table_plane": list(plane)} if plane is not None and not a.no_render_table else None
    if geo:
        log(f"render: {TB.describe(plane)} (scene geometry override, evaluation only)")
    views = VI.make_views(st.cfg, a.render_worlds, seed=a.seed, allow_change=a.allow_scene_change, device=a.device,
                          geometry_over=geo)
    r = evaluate(st, jt, t, a.world, cache, views, a.randomise, a.jaw_lead, a.record_states, a.full_horizon, a.device,
                 lag=a.lag, lag_range=a.lag_range, qd_window=a.qd_robot, light_redraw_p=a.light_redraw_p)
    r["student"] = np.array(str(a.student))
    if a.out:
        np.savez_compressed(a.out, **r)
    log(f"student {a.world}{' randomised' if a.randomise else ''}: {str(r['summary'])}")


def cmd_collect(a):
    jt = JE.JointTeacher(x(a.jt), a.device)
    t = load_tasks(a.tasks, np.arange(a.first, a.first + a.n) if a.n else None)
    n = len(t["seeds"])
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon, hold_cap_s=a.hold_cap_s)
    rng = np.random.default_rng(a.seed)
    student = None
    src = np.zeros(n, np.int64)
    if a.student:
        st, _ = VI.load_student(a.student, a.device)
        views = VI.make_views(st.cfg, a.render_worlds, seed=a.seed, allow_change=a.allow_scene_change, device=a.device)
        student = VI.Actor(st, views, randomise=a.randomise, device=a.device, lag_range=st.cfg.lag_range, seed=a.seed)
        src = np.where(rng.random(n) < a.teacher_share, 0, 1)
    noise = np.where((src == 0) & (rng.random(n) < a.noise_frac), a.noise, 0.0)
    kmax = max(int(a.lag_kmax), st.cfg.lag_max if a.student else 0)   # a lag above viral.LAG_REC needs more states
    d = collect(t, a.world, jt, cache, src, noise, a.seed, a.batch, student, lag_rec=a.lag_rec or a.lag_kmax > 0,
                lag_kmax=kmax)
    np.savez_compressed(a.out, **d)
    log(f"-> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--device", default="cuda")
        q.add_argument("--nrd-ckpt", default=NRD)
        q.add_argument("--jt", required=True)
        q.add_argument("--mj-python", default=MJ_PY)
        q.add_argument("--horizon", type=int, default=400)
        q.add_argument("--hold-cap-s", type=float, default=None,
                       help="evaluation / collection envs: the hold_cap rule (s held in the air; default 3.5)")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--render-worlds", type=int, default=256)
        q.add_argument("--allow-scene-change", action="store_true",
                       help="real scene: render with scene / calibration files whose sha256 differ from the student's")

    q = sub.add_parser("run")
    common(q)
    q.add_argument("--out", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--eval-tasks", nargs="+", required=True)
    q.add_argument("--eval-n", type=int, default=0)
    q.add_argument("--guard-tasks", nargs="*", default=["~/so101_policy/runs/viral/tasks/fresh100.npz"])
    q.add_argument("--config", required=True)
    q.add_argument("--set", nargs="*")
    q.add_argument("--world", default="nrd")
    q.add_argument("--r0-tasks", type=int, default=2048)
    q.add_argument("--rounds", type=int, default=4)
    q.add_argument("--round-tasks", type=int, default=1024)
    q.add_argument("--teacher-share", type=float, default=0.5)
    q.add_argument("--r0-noise-frac", type=float, default=0.5)
    q.add_argument("--noise", type=float, default=1.0)
    q.add_argument("--collect-randomise", action="store_true")
    q.add_argument("--batch", type=int, default=1024)
    q.add_argument("--train-steps0", type=int, default=8000)
    q.add_argument("--train-steps", type=int, default=4000)
    q.add_argument("--init-from", default="")
    q.add_argument("--extra-data", nargs="*")
    q.add_argument("--eval-worlds", nargs="+", default=["mujoco"])
    q.add_argument("--eval-randomise", action="store_true")
    q.add_argument("--eval-every", type=int, default=1)
    q.add_argument("--eval-workers", type=int, default=8)
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("online")
    common(q)
    q.add_argument("--out", required=True)
    q.add_argument("--init", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--eval-tasks", nargs="+", required=True)
    q.add_argument("--eval-n", type=int, default=0)
    q.add_argument("--guard-tasks", nargs="*", default=["~/so101_policy/runs/viral/tasks/fresh100.npz"])
    q.add_argument("--config", default="")
    q.add_argument("--set", nargs="*")
    q.add_argument("--online-config", default="")
    q.add_argument("--oset", nargs="*")
    q.add_argument("--eval-randomise", action="store_true")
    q.add_argument("--eval-workers", type=int, default=8)
    q.set_defaults(fn=cmd_online)

    q = sub.add_parser("eval")
    common(q)
    q.add_argument("--student", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--world", default="mujoco")
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--randomise", action="store_true")
    q.add_argument("--jaw-lead", type=int, default=None)
    q.add_argument("--record-states", action="store_true")
    q.add_argument("--full-horizon", action="store_true")
    q.add_argument("--lag", type=int, default=None, help="image lag (control steps); default the student's cfg.lag")
    q.add_argument("--lag-range", type=int, nargs=2, default=None, help="image lag drawn per world (lo hi)")
    q.add_argument("--qd-robot", type=int, default=0,
                   help="qdot input = the robot's slope over this many 20 ms states of the quantised q (0 = sim qdot)")
    q.add_argument("--goal-shift", default="",
                   help="evaluation-only goal shift (overshoot gate), e.g. 'p=1,d=10:25,lat=8' (mm; JE.GoalJumpConfig "
                        "with the training trigger; defaults n=1 jump per episode, before=0: the cube ends PAST the "
                        "moved goal); the student's renders and the success check use the moved goal")
    q.add_argument("--light-redraw-p", type=float, default=1.0,
                   help="with --randomise: per decision the probability of a new random light / intrinsics draw, else "
                        "the last one stays (an image-history student: use its online config's light_redraw_p); "
                        "1 = a new draw every decision")
    q.add_argument("--out", default="")
    q.add_argument("--scene-path", default="", help="the student's scene file at this path (same sha256)")
    q.add_argument("--calib-path", default="", help="the student's calibration file at this path (same sha256)")
    from nedm.so101_policy import table as TB
    TB.add_cli(q, student=True)
    q.set_defaults(fn=cmd_eval)

    q = sub.add_parser("collect")
    common(q)
    q.add_argument("--tasks", required=True)
    q.add_argument("--world", default="nrd")
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--student", default="")
    q.add_argument("--teacher-share", type=float, default=0.5)
    q.add_argument("--noise-frac", type=float, default=0.5)
    q.add_argument("--noise", type=float, default=1.0)
    q.add_argument("--randomise", action="store_true")
    q.add_argument("--batch", type=int, default=1024)
    q.add_argument("--lag-rec", action="store_true", help="record the 20 ms states for image lag (row_lag)")
    q.add_argument("--lag-kmax", type=int, default=0,
                   help="row_lag states per row (default viral.LAG_REC 4 or the student lag_max; implies --lag-rec)")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_collect)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
