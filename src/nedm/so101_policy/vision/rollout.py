"""Closed-loop evaluation of the camera student in any world (CPU MuJoCo = the ground truth, the NRD, the kinematic
world), with the episode loop of ``vec_env.run_episodes`` and the evaluation env of ``demos.make_env`` (the decoder
config is the student's own, stored with its demos: review R1).

Per decision: the world's state (true cube orientation) and the goal plate are drawn by ``dataset.Views`` (Madrona on
the GPU; CPU MuJoCo states are moved to the GPU), the student predicts a chunk of decoder actions from the images,
the proprio (q + ``DecoderV2.observation()``) and the goal vector, all worlds in one batch; temporal ensembling (ACT's
weights, the student's coefficient) over the predictions made since the decoder mode last changed (``ModeEnsembler``;
plain ``ACTTemporalEnsembler`` with ``ensemble="act"``) gives the executed action, clipped to [-1, 1]; except the trigger
a[4] (``trigger_latest``, default): it comes from the newest chunk's first action. Reason: the expert's PLACE trigger is
a one-decision impulse whose time varies with the CARRY line's random duration; averaging up to 20 predictions of it
(nearly uniform weights at coefficient 0.01) smears it below the threshold and delays the PLACE, and a smeared GRASP
trigger inside the descent would abort the GRASP (a[4] <= 0 aborts before the close). The output has
the ``eval_policy.py run`` format (per-episode outcome, per-step records), so ``eval_policy.py report`` compares runs.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import demos as D


class ModeEnsembler:
    """Temporal ensembling per world (ACT Algorithm 2: weights exp(-coeff i), i = 0 the oldest prediction) over the
    predictions made since the decoder mode last changed. Reason: the decoder's macros have random durations the
    student cannot see, so a prediction made inside a GRASP macro for "k decisions later" mixes macro decisions (expert
    labels 0) with the first decision in hold (the CARRY target); averaging it into that decision shrinks the CARRY
    line. With no mode change in the window this equals ``ACTTemporalEnsembler`` (unit test)."""

    def __init__(self, coeff, chunk):
        self.coeff, self.chunk = float(coeff), int(chunk)
        self.buf = self.valid = self.mode = None

    def reset(self):
        self.buf = self.valid = self.mode = None

    def update(self, chunk, mode):
        """chunk [B, C, A] (predicted now), mode [B] (decoder mode now) -> action [B, A] for now."""
        B, C, A = chunk.shape
        dev = chunk.device
        if self.buf is None:
            self.buf = torch.zeros(B, C, C, A, device=dev, dtype=chunk.dtype)     # [world, age j, offset, dim]
            self.valid = torch.zeros(B, C, dtype=torch.bool, device=dev)
            self.mode = mode.clone()
        self.buf = torch.roll(self.buf, 1, dims=1)
        self.valid = torch.roll(self.valid, 1, dims=1)
        self.valid[:, 0] = False                       # the slot that wrapped around (age C) is too old
        changed = mode != self.mode
        self.valid[changed] = False
        self.mode = mode.clone()
        self.buf[:, 0] = chunk
        self.valid[:, 0] = True
        j = torch.arange(C, device=dev)
        pred = self.buf[:, j, j]                       # [B, C(age), A]: the prediction for now made j decisions ago
        n = self.valid.sum(1, keepdim=True)            # valid ages form a prefix 0..n-1
        w = torch.exp(-self.coeff * (n - 1 - j[None]).clamp_min(0).to(chunk.dtype)) * self.valid.to(chunk.dtype)
        return (pred * w[..., None]).sum(1) / w.sum(1, keepdim=True)


class StudentAct:
    """act_fn of ``vec_env.run_episodes`` for a ``student.Student`` (module docstring). ``ensemble``: "mode" (default,
    ``ModeEnsembler``), "act" (LeRobot's ``ACTTemporalEnsembler``, plain ACT), "none" (the newest chunk's first action).
    ``chunk_exec`` > 0: open-loop execution of chunk_exec actions per model call (ablation). ``trigger_latest``: a[4]
    from the newest chunk (module docstring)."""

    def __init__(self, student, views, device="cuda", ensemble="mode", chunk_exec=0, trigger_latest=True,
                 trigger_thr=None):
        if ensemble is True:
            ensemble = "mode"
        elif ensemble is False:
            ensemble = "none"
        if ensemble not in ("mode", "act", "none"):
            raise ValueError(f"ensemble {ensemble!r}: mode, act or none")
        self.st, self.views, self.dev = student, views, torch.device(device)
        self.ensemble, self.chunk_exec, self.trigger_latest = ensemble, int(chunk_exec), bool(trigger_latest)
        # execution thresholds of the predicted trigger (GRASP side: mode free and the pre-close GRASP phases; PLACE
        # side: mode hold): a[4] is sent as +1 above the threshold, else -1; None = the decoder's own rule (> 0)
        self.trigger_thr = None if trigger_thr is None else (float(trigger_thr[0]), float(trigger_thr[1]))
        self.ens, self.queue, self.k = None, None, 0
        self.t_render = self.t_model = 0.0
        self.calls = 0

    def reset(self, env):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        c = self.st.cfg
        self.ens = (ACTTemporalEnsembler(c.ensemble_coeff, c.chunk) if self.ensemble == "act" else
                    ModeEnsembler(c.ensemble_coeff, c.chunk) if self.ensemble == "mode" else None)
        self.queue, self.k = None, 0
        self.goal_pose = torch.cat((env.task.goal_xy, env.task.goal_yaw[:, None]), -1).to(self.dev, torch.float32)
        self.goal = V.goal_vec(env.task.goal_xy.cpu(), env.task.goal_yaw.cpu()).to(self.dev)

    @torch.no_grad()
    def __call__(self, obs, env):
        ws = env.ws
        if self.chunk_exec and self.queue is not None and self.k % self.chunk_exec != 0:
            a = self.queue[:, self.k % self.chunk_exec]
            self.k += 1
            return a.to(env.dev, torch.float64).clamp(-1, 1), True
        t0 = time.time()
        q = ws.q.to(self.dev, torch.float32)
        rgb = self.views.render(q, ws.cube_pos.to(self.dev, torch.float32), ws.cube_quat.to(self.dev, torch.float32),
                                self.goal_pose)
        img = V.to_input(rgb)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        state = V.proprio(q, env.dec.observation().to(self.dev, torch.float32))
        goal = self.goal
        if getattr(self.st.cfg, "privileged_cube", False):        # ablation: the true cube state
            goal = torch.cat((goal, V.cube_vec(ws.cube_pos.to(self.dev), ws.cube_quat.to(self.dev)).to(self.dev)), -1)
        with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=bool(self.st.cfg.amp)):
            chunk = self.st.predict_chunk(img, state, goal)
        chunk = chunk.float()
        if self.ens is not None and not self.chunk_exec:
            a = (self.ens.update(chunk, env.dec.mode.to(self.dev)) if self.ensemble == "mode" else self.ens.update(chunk))
            if self.trigger_latest:
                a = a.clone()
                a[:, 4] = chunk[:, 0, 4]
        else:
            self.queue = chunk[:, : max(1, self.chunk_exec)]
            a = self.queue[:, 0]
        if self.trigger_thr is not None:
            from nedm.so101_policy.decoder import M
            mode = env.dec.mode.to(self.dev)
            grasp_side = (mode == M["free"]) | ((mode >= M["g_brake"]) & (mode <= M["g_settle"]))
            thr = torch.where(grasp_side, self.trigger_thr[0], self.trigger_thr[1]).to(a.dtype)
            a = a.clone()
            a[:, 4] = torch.where(a[:, 4] > thr, 1.0, -1.0)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.t_render += t1 - t0
        self.t_model += time.time() - t1
        self.calls += 1
        self.k += 1
        return a.to(env.dev, torch.float64).clamp(-1, 1), True


@torch.no_grad()
def evaluate(student, tasks, world, views, ckpt=None, device="cuda", workers=0, mj_python=None, horizon=400,
             record=True, members="split", ensemble="mode", chunk_exec=0, log=print, policy_path="", trigger_latest=True,
             trigger_thr=None):
    """The student on all tasks in one world -> dict of numpy arrays (``eval_policy.py run`` format + timing)."""
    from nedm.so101_policy.vec_env import run_episodes
    B = len(tasks["seeds"])
    t0 = time.time()
    env = D.make_env(world, B, device, ckpt, workers, mj_python, student.decoder_cfg, horizon, members, record)
    dev = env.dev
    act = StudentAct(student, views, device, ensemble, chunk_exec, trigger_latest, trigger_thr)
    g = lambda k: torch.as_tensor(tasks[k], dtype=torch.float64, device=dev)              # noqa: E731
    mj_start = dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"], cube_quat=tasks["start_cube_quat"])
    r = run_episodes(env, act, D.start_state(tasks, dev), g("goal_xy"), g("goal_yaw"),
                     torch.as_tensor(tasks["seeds"], dtype=torch.long, device=dev),
                     mj_start=mj_start if world == "mujoco" else None, reset_hook=act.reset, log=log)
    rec = env.records() if record else {}
    resolved = env.resolved() if hasattr(env, "resolved") else {}
    steps = env.world_steps
    env.close()
    out = dict(rec)
    out.update(end=r["end"], success=r["success"], success10=r["success10"], lifted=r["lifted"], end_xy=r["end_xy"],
               end_yaw=r["end_yaw"], t_end=r["steps"], ret=r["ret"], length=r["length"], seeds=np.asarray(tasks["seeds"]),
               box_log=np.zeros(B, bool), world=np.array(world), config=np.array("vision"), decoder=np.array("v2"),
               ckpt=np.array(str(ckpt)), wall_s=np.float64(time.time() - t0), device=np.array(str(dev)),
               world_steps=np.int64(steps), phase=np.zeros_like(rec["mode"]) if record else np.zeros(0),
               label=np.zeros_like(rec["mode"]) if record else np.zeros(0), phase_names=np.array(["policy"]),
               train_rules=np.bool_(False), assist=np.bool_(False), stochastic=np.bool_(False), noise_seed=np.int64(0),
               policy=np.array(str(policy_path)), env_resolved=np.array(json.dumps(resolved, default=str)),
               render_s=np.float64(act.t_render), model_s=np.float64(act.t_model), model_calls=np.int64(act.calls),
               ensemble=np.array(str(ensemble)), chunk_exec=np.int64(chunk_exec), trigger_latest=np.bool_(trigger_latest),
               trigger_thr=np.array([np.nan, np.nan] if trigger_thr is None else trigger_thr, dtype=np.float64))
    return out


# ---- DAgger: the student drives, the FSM expert labels every visited state (spec_v2 B7) ---------------------------------
HARNESS_GRASP_TOL = (0.0045, 0.010)        # m: gate_expert.GRASP_TOL (|x| closing axis, |y| jaw width)
HARNESS_GRASP_YAW = math.radians(2.5)       # gate_expert.GRASP_YAW
HARNESS_EXPERT_CFG = dict(abort_off=0.013, yaw_ok=HARNESS_GRASP_YAW)   # gate_expert.EXPERT_CFG["align"]


class ShadowExpert:
    """The FSM expert run alongside another policy: ``label(env)`` -> the expert's decoder action for the CURRENT
    state. The expert's phase machine follows the decoder mode; where the driving policy put the decoder in a mode the
    expert's phase does not expect (the student started a GRASP the expert would not have started, a PLACE, ...), the
    expert is re-synchronised with ``PickPlaceExpert.reset`` (phase inferred from the decoder mode and the state: GRASP
    macro -> grasp (it then holds the trigger, or aborts if the cube is > ``abort_off`` off the descent axis), hold with
    the cube held -> carry, hold without it -> recover, PLACE macro -> place).

    ``harness`` (default True): the GRASP trigger of the gate's noisy-alignment harness (gate_expert.md sec. 5.1): in
    the approach, with the trigger armed, the jaw open and the cube at rest, the label is a GRASP trigger as soon as the
    decoder's brake end (``DecoderV2.brake_stop`` = ``macro_goal``) has the cube within |x| 4.5 mm, |y| 10 mm and 2.5 deg
    (MuJoCo: 95.5 % success with such triggers in the gate), and the expert runs with ``HARNESS_EXPERT_CFG`` (abort only
    beyond 13 mm). Reason (measured 2026-10-12 00:45): with the expert's own trigger (1.5 mm / 1 deg / 1 cm/s) the
    expert labelled 0-8 GRASP triggers in 27k-37k student states; DAgger then taught the NRD-arm student to stop
    triggering (GRASP starts 88 -> 16 %, success 14.5 -> 3.1 %)."""

    def __init__(self, cfg=None, harness=True):
        base = dict(HARNESS_EXPERT_CFG) if harness else {}
        base.update(dict(cfg or {}))
        self.cfg, self.harness, self.ex = base, bool(harness), None
        self.resyncs = self.harness_triggers = 0

    def reset(self, env):
        from dataclasses import replace

        from nedm.so101_policy.expert import ExpertConfig, PickPlaceExpert
        self.ex = PickPlaceExpert(env.dec, replace(ExpertConfig(), **self.cfg))
        self.ex.reset(torch.arange(env.B, device=env.dev), env.ws, env.task.goal_xy.clone(), env.task.goal_yaw.clone())

    @torch.no_grad()
    def label(self, env):
        from nedm.so101_policy.decoder import M
        from nedm.so101_policy.expert import PH
        ex, dec, ws = self.ex, env.dec, env.ws
        mode, ph = dec.mode, ex.phase
        in_grasp = (mode >= M["g_brake"]) & (mode <= M["g_lift"])
        hold, place = mode == M["hold"], mode >= M["p_brake"]
        bad = ((in_grasp & (ph != PH["grasp"])) | (hold & (ph != PH["carry"]) & (ph != PH["recover"]) & (ph != PH["grasp"]))
               | (place & (ph != PH["place"]) & (ph != PH["recover"]) & (ph != PH["carry"])))
        idx = bad.nonzero().flatten()
        if len(idx):
            self.resyncs += len(idx)
            ex.reset(idx, ws.index(idx), env.task.goal_xy[idx].clone(), env.task.goal_yaw[idx].clone())
        a = ex.act(ws)
        if self.harness:
            from nedm.so101_policy.task import cube_yaw_tilt, wrap90
            approach = (ex.phase == PH["approach"]) & (mode == M["free"])
            jaw_ready = (dec.jaw_k >= dec.jaw_n) & ((dec.jaw_to - dec.cfg.jaw_open).abs() < 1e-9)
            stop, syaw = dec.brake_stop()
            cube = ws.cube_pos.to(torch.float64)
            cyaw, _ = cube_yaw_tilt(ws.cube_quat.to(torch.float64))
            d = cube[:, :2] - stop[:, :2]
            c, s_ = torch.cos(syaw), torch.sin(syaw)
            x, y = c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]
            inbox = ((x.abs() <= HARNESS_GRASP_TOL[0]) & (y.abs() <= HARNESS_GRASP_TOL[1])
                     & (wrap90(syaw - cyaw).abs() <= HARNESS_GRASP_YAW))
            ok = (approach & dec.armed & jaw_ready & (a[:, 4] <= 0) & inbox
                  & (ws.cube_v.to(torch.float64).norm(dim=-1) < ex.cfg.cube_still))
            self.harness_triggers += int(ok.sum())
            a[:, 4] = torch.where(ok, 1.0, a[:, 4])
            ex.phase = torch.where(ok, PH["grasp"], ex.phase)
        return a


class DaggerAct:
    """act_fn: the student acts (``StudentAct``), the shadow expert labels (``last_label``); with probability ``beta``
    per world and decision (counter-based hash of the task seed and the decision index) the expert's action is executed
    instead (beta-mixing). ``demos.DecisionRecorder`` stores ``last_label`` as the action and the executed one as
    ``executed``."""

    def __init__(self, student_act, expert_cfg=None, beta=0.0, seed=0, harness=True):
        self.sa, self.sh, self.beta, self.seed = student_act, ShadowExpert(expert_cfg, harness), float(beta), int(seed)
        self.k, self.last_label, self.seeds = 0, None, None

    def reset(self, env):
        self.sa.reset(env)
        self.sh.reset(env)
        self.k = 0
        self.seeds = env.task_seed.clone()

    def __call__(self, obs, env):
        lab = self.sh.label(env)
        a, _ = self.sa(obs, env)
        if self.beta > 0:
            h = (self.seeds * 0x9E3779B1 + self.k * 0x85EBCA6B + self.seed * 0xC2B2AE35) & 0xFFFFFFFF
            u = ((h * 2654435761) & 0xFFFFFFFF).double() / 2 ** 32
            a = torch.where((u < self.beta)[:, None], lab.to(a), a)
        self.k += 1
        self.last_label = lab.clamp(-1.0, 1.0)
        return a, True


class NoisyExpertAct:
    """DART-style demonstrations: the FSM expert (``ShadowExpert``, harness GRASP trigger) drives with pre-tanh Gaussian
    noise on its free-motion action a[0:4] (a' = tanh(atanh(a) + sigma eps), the gate's noise; counter-based normals of
    (task seed, decision index): the same noise in every world) on the decisions of its approach phase (mode free);
    the label (``last_label``) is the expert's clean action at the reached state. Macros, the carry and the triggers
    get no noise. ``DecisionRecorder`` stores label and executed action (meta ``dagger``: every episode is kept)."""

    def __init__(self, sigma=0.3, expert_cfg=None, seed=0, harness=True):
        self.sh, self.sigma, self.seed = ShadowExpert(expert_cfg, harness), float(sigma), int(seed)
        self.k, self.last_label, self.seeds = 0, None, None

    def reset(self, env):
        self.sh.reset(env)
        self.k = 0
        self.seeds = env.task_seed.clone() * 1_000_003 + 7919 * self.seed

    def __call__(self, obs, env):
        from nedm.so101_policy.decoder import M, hash_uniform
        from nedm.so101_policy.expert import PH
        lab = self.sh.label(env)
        a = lab.clone()
        noisy = (self.sh.ex.phase == PH["approach"]) & (env.dec.mode == M["free"]) & (lab[:, 4] <= 0)
        if self.sigma > 0 and bool(noisy.any()):
            cnt = torch.full_like(self.seeds, self.k)
            eps = torch.stack([torch.sqrt(-2.0 * torch.log1p(-hash_uniform(self.seeds, cnt, 2 * j + 1)))
                               * torch.cos(2 * math.pi * hash_uniform(self.seeds, cnt, 2 * j + 2)) for j in range(4)], -1)
            x = torch.tanh(torch.atanh(a[:, :4].clamp(-0.999, 0.999)) + self.sigma * eps.to(a))
            a[:, :4] = torch.where(noisy[:, None], x, a[:, :4])
        self.k += 1
        self.last_label = lab.clamp(-1.0, 1.0)
        return a, True


@torch.no_grad()
def collect_noisy_expert(tasks, world, ckpt=None, device="cuda", workers=0, mj_python=None, horizon=400, sigma=0.3,
                         expert_cfg=None, seed=0, decoder_cfg=None, harness=True, log=print, batch=0):
    """DART demonstrations (``NoisyExpertAct``) on all tasks -> packed demo dict (meta ``dagger`` True, ``dart``)."""
    from nedm.so101_policy.vec_env import run_episodes
    n = len(tasks["seeds"])
    batch = int(batch) or n
    t0, steps, htrig, parts, code = time.time(), 0, 0, [], D.source_hashes()
    dec_cfg = dict(D.DECODER_CFG if decoder_cfg is None else decoder_cfg)
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        t = {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}
        env = D.make_env(world, len(idx), device, ckpt, workers, mj_python, dec_cfg, horizon)
        dev = env.dev
        act = NoisyExpertAct(sigma, expert_cfg, seed, harness)
        rec = D.DecisionRecorder(act)
        g = lambda k: torch.as_tensor(t[k], dtype=torch.float64, device=dev)              # noqa: E731
        mj_start = dict(q=t["start_q"], cube_pos=t["start_cube_pos"], cube_quat=t["start_cube_quat"])
        ep = run_episodes(env, rec, D.start_state(t, dev), g("goal_xy"), g("goal_yaw"),
                          torch.as_tensor(t["seeds"], dtype=torch.long, device=dev),
                          mj_start=mj_start if world == "mujoco" else None, reset_hook=rec.reset, log=None)
        steps += env.world_steps
        htrig += int(act.sh.harness_triggers)
        ex_cfg = dict(act.sh.cfg)
        env.close()
        parts.append(D.pack(rec.arrays(), ep, t))
        log(f"  noisy expert {world}: tasks {s0}-{s0 + len(idx) - 1}, success {ep['success'].mean() * 100:.1f} %, "
            f"{time.time() - t0:.0f} s")
    out = D.concat(parts)
    meta = dict(world=world, ckpt=str(ckpt), n_tasks=n, wall_s=time.time() - t0, world_steps=int(steps),
                decoder_cfg=dec_cfg, expert_cfg_resolved=ex_cfg, horizon=int(horizon), dagger=True, dart=True,
                sigma=sigma, harness=bool(harness), harness_triggers=htrig, success=float(np.mean(out["ep_success"])),
                decisions=int(len(out["t"])), code=code)
    out["meta"] = np.array(json.dumps(meta))
    return out


@torch.no_grad()
def collect_dagger(student, tasks, world, views, ckpt=None, device="cuda", workers=0, mj_python=None, horizon=400,
                   beta=0.0, expert_cfg=None, seed=0, log=print, batch=256, harness=True):
    """Student rollouts with expert labels on all tasks -> packed demo dict (``demos.pack``; ``action`` = the expert's
    label, ``executed`` = the action sent; meta ``dagger`` True: the dataset keeps every episode, failed ones too: an
    episode ends at the first task-rule violation, so every stored state precedes it). ``batch`` worlds per env (GPU
    memory: the NRD world, the renderer and ACT share the 5090 with other jobs); views must hold >= batch worlds."""
    from nedm.so101_policy.vec_env import run_episodes
    n = len(tasks["seeds"])
    batch = int(batch) or n
    t0, steps, resyncs, htrig, parts = time.time(), 0, 0, 0, []
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        t = {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}
        B = len(idx)
        env = D.make_env(world, B, device, ckpt, workers, mj_python, student.decoder_cfg, horizon)
        dev = env.dev
        act = DaggerAct(StudentAct(student, views, device), expert_cfg, beta, seed, harness)
        rec = D.DecisionRecorder(act)
        g = lambda k: torch.as_tensor(t[k], dtype=torch.float64, device=dev)              # noqa: E731
        mj_start = dict(q=t["start_q"], cube_pos=t["start_cube_pos"], cube_quat=t["start_cube_quat"])
        ep = run_episodes(env, rec, D.start_state(t, dev), g("goal_xy"), g("goal_yaw"),
                          torch.as_tensor(t["seeds"], dtype=torch.long, device=dev),
                          mj_start=mj_start if world == "mujoco" else None, reset_hook=rec.reset, log=None)
        steps += env.world_steps
        resyncs += int(act.sh.resyncs)
        htrig += int(act.sh.harness_triggers)
        ex_cfg = dict(act.sh.cfg)
        env.close()
        parts.append(D.pack(rec.arrays(), ep, t))
        del env, act, rec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = D.concat(parts)
    meta = dict(world=world, ckpt=str(ckpt), n_tasks=n, wall_s=time.time() - t0, world_steps=int(steps),
                decoder_cfg=dict(student.decoder_cfg), expert_cfg=dict(expert_cfg or {}), horizon=int(horizon),
                dagger=True, beta=beta, resyncs=resyncs, harness=bool(harness), harness_triggers=htrig,
                expert_cfg_resolved=ex_cfg, success=float(np.mean(out["ep_success"])),
                decisions=int(len(out["t"])), batch=batch, code=D.source_hashes())
    out["meta"] = np.array(json.dumps(meta))
    log(f"  dagger {world}: {n} tasks, student success {meta['success'] * 100:.1f} %, resyncs {meta['resyncs']}, "
        f"{meta['wall_s']:.0f} s")
    return out
