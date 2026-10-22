"""The Markov joint teacher (JT) of build9: behaviour cloning from the current teacher + decoder v3, PPO fine-tune in the
NRD, and fixed-task episodes in any world (``joint_env`` module docstring for the observation, action and action layer).

1. ``expert_collect``: the current teacher (state PPO through decoder v3 = build8's ``tray.TEACHER_DECODER``) in a
   ``tray.TrayWorld`` (label decoder + DART driver decoder: labels = the CLEAN teacher's joint targets from the noisy
   state). Recorded per decision boundary: the JT observation (``joint_env.markov_obs``, 47) next to build8's row
   records; per 20 ms step the label targets ``u_lab``. Tasks: the start bank (production_plus split 0, as PPO) +
   ``task.sample_goals`` (``bank_tasks``; decoder / noise seeds 700,000+), or any task file.
2. ``BCSet`` / ``train_bc``: rows (obs, the 5 label targets of the decision relative to q) -> a ``ppo.Policy`` whose actor
   outputs the normalised coefficients (``joint_env.Codec`` fitted on the data); loss = L1 on the 5 reconstructed targets
   divided by a per-joint scale (std of the relative targets), masked inside the episode.
3. ``JointPPO``: ``ppo.PPO`` with (a) ``critic_warmup`` iterations in which the actor and its std do not change (the
   critic is fitted first: a BC actor with a random critic would be pushed by noise advantages), (a2) the observation
   normaliser frozen at the BC statistics (measured: PPO's running update moved the frozen BC actor's NRD development
   success from 95 to 12 % in 20 warm-up iterations), (b) an optional
   behaviour-cloning anchor ``bc_coef`` x mean squared distance of the actor mean to a frozen reference actor on the
   rollout states, (c) the lenient success in the learning curve.
4. ``run_tasks``: one episode per task in a ``JointVecEnv`` (auto reset off) driven per WORLD by the JT (with optional
   noise on its action) or by a student callback; -> outcomes and (``record``) per-boundary rows for student data.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import ppo as P
from nedm.so101_policy import task as T

BANK_SEED0 = 700_000


def log_default(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ---- tasks from the start bank ----------------------------------------------------------------------------------------
def bank_tasks(n, seed=0, first=0, starts=str(JE.PP_DATA), split=0, device="cpu"):
    """n tasks from the start bank (``vec_env.load_starts``) with goals from ``task.sample_goals`` (generator ``seed``):
    the tray-task layout with state_* only (no MuJoCo start: NRD / kinematic worlds). Seeds BANK_SEED0 + first + i
    (they seed the decoder durations and the noise only)."""
    from nedm.so101_policy.vec_env import load_starts
    bank = load_starts(starts, split, device)
    g = torch.Generator(device).manual_seed(int(seed))
    i = torch.randint(bank["q"].shape[0], (int(n),), generator=g, device=device)
    yaw, _ = T.cube_yaw_tilt(bank["cube_quat"][i])
    gxy, gyaw = T.sample_goals(bank["cube_pos"][i, :2], yaw, g)
    out = dict(seeds=np.arange(n, dtype=np.int64) + BANK_SEED0 + int(first), goal_xy=gxy.cpu().numpy(),
               goal_yaw=gyaw.cpu().numpy(), bank_index=i.cpu().numpy())
    for k in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w"):
        out[f"state_{k}"] = bank[k][i].cpu().numpy()
    return out


# ---- 1. expert data ---------------------------------------------------------------------------------------------------
def expert_world_class():
    from nedm.so101_policy.vision import tray as TR

    class ExpertWorld(TR.TrayWorld):
        """``tray.TrayWorld`` that also records the JT observation at every decision boundary and the lenient flag."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            JE.attach_lenient(self.task)
            self._mobs = None

        def observation(self, F):
            if self._mobs is not None:
                self._mobs.append(JE.markov_obs(F, self.task).cpu())
            return super().observation(F)

        def run_expert(self, tasks, noise=None, seed=0, record=True, record_states=False):
            self._mobs = []
            r = self.run(tasks, None, 1.0, noise, seed, record=record, record_states=record_states)
            M = torch.stack(self._mobs, 1).numpy()
            self._mobs = None
            if record:
                r["row_mobs"] = M[r["row_ep"], r["row_t"].astype(np.int64) // JE.STEPS]
            r["lenient"] = self.task.lenient.cpu().numpy()
            r["lenient_t"] = self.task.lenient_t.cpu().numpy()
            return r

    return ExpertWorld


@torch.no_grad()
def expert_collect(tasks, teacher, world="nrd", noise=0.0, seed=0, device="cuda", nrd_ckpt=None, workers=0,
                   mj_python=None, horizon=400, batch=1024, log=log_default, record=True, cache=None):
    """The current teacher + decoder v3 on all tasks (batches of ``batch`` worlds) -> packed dict (build8 layout +
    row_mobs, lenient) + meta. noise: float or [n] (DART scale of the teacher's move columns per task)."""
    from nedm.so101_policy.vision import tray as TR
    EW = expert_world_class()
    n = len(tasks["seeds"])
    nzs = np.broadcast_to(np.asarray(noise, np.float64), (n,))
    parts, t0 = [], time.time()
    cache = {} if cache is None else cache
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        t = TR.subset(tasks, idx)
        key = (world, len(idx))
        if key not in cache:
            cache[key] = EW(world, len(idx), teacher, device, nrd_ckpt, workers, mj_python, horizon)
        tw = cache[key]
        r = tw.run_expert(t, nzs[idx], seed, record=record)
        parts.append(r)
        log(f"  expert {world}: tasks {s0}-{idx[-1]}, success {r['success'].mean() * 100:.1f} %, lenient "
            f"{r['lenient'].mean() * 100:.1f} %, {time.time() - t0:.0f} s")
    out = TR.concat(parts) if record else {k: np.concatenate([p[k] for p in parts]) for k in
                                           ("success", "lenient", "end", "lifted", "end_xy", "end_yaw", "steps")}
    if record:
        out["lenient"] = np.concatenate([p["lenient"] for p in parts])
    meta = dict(world=world, nrd_ckpt=str(nrd_ckpt), teacher=teacher.info(), n_tasks=n, noise=float(np.mean(nzs)),
                seed=int(seed), horizon=int(horizon), wall_s=time.time() - t0, success=float(np.mean(out["success"])),
                lenient=float(np.mean(out["lenient"])), seeds=[int(tasks["seeds"][0]), int(tasks["seeds"][-1])],
                decoder_over=TR.TEACHER_DECODER)
    out["meta"] = np.array(json.dumps(meta, default=str))
    return out


# ---- 2. behaviour cloning -------------------------------------------------------------------------------------------
class BCSet:
    """Expert files -> tensors on ``device``: obs [N, 47], rel [N, 5, 6] (label targets - q at the boundary), valid
    [N, 5]. Validation = ``val_frac`` of the episodes of the FIRST file (seeded)."""

    def __init__(self, paths, device="cuda", val_frac=0.03, seed=0):
        obs, rel, val, ok = [], [], [], []
        for fi, p in enumerate(paths):
            z = np.load(Path(p).expanduser())
            ep, t = z["row_ep"].astype(np.int64), z["row_t"].astype(np.int64)
            s0, st = z["ep_step_start"].astype(np.int64), z["steps"].astype(np.int64)
            j = np.arange(JE.STEPS)
            g = s0[ep][:, None] + t[:, None] + j[None]
            v = (t[:, None] + j[None]) < st[ep][:, None]
            g = np.minimum(g, len(z["u_lab"]) - 1)
            rel.append(z["u_lab"][g] - z["row_q"][:, None, :])
            obs.append(z["row_mobs"])
            ok.append(v)
            E = len(st)
            isv = np.zeros(E, bool)
            if fi == 0:
                rng = np.random.default_rng(seed)
                isv[rng.permutation(E)[: int(round(val_frac * E))]] = True
            val.append(isv[ep])
        tt = lambda x, dt=torch.float32: torch.as_tensor(np.concatenate(x), dtype=dt, device=device)   # noqa: E731
        self.obs, self.rel, self.valid = tt(obs), tt(rel), tt(ok, torch.bool)
        isval = tt(val, torch.bool)
        self.train_rows = torch.nonzero(~isval).flatten()
        self.val_rows = torch.nonzero(isval).flatten()
        self.dev = torch.device(device)

    def codec(self):
        full = self.valid[self.train_rows].all(1)
        return JE.Codec.fit(self.rel[self.train_rows][full].double().cpu())

    def scale(self, floor=2e-3):
        v = self.valid[self.train_rows][..., None].float()
        r = self.rel[self.train_rows]
        m = (r * v).sum((0, 1)) / v.sum((0, 1))
        return (((r - m) ** 2 * v).sum((0, 1)) / v.sum((0, 1))).sqrt().clamp_min(floor)          # [6]


RATE_DIMS = list(range(6, 12)) + list(range(27, 33))     # qd, cube v, cube w in the 47-dim observation


def bc_loss(pol, codec: JE.Codec, obs, rel, valid, scale, rate_noise=0.0):
    """L1 on the 5 reconstructed targets / per-joint scale. ``rate_noise``: Gaussian noise (std units) on the normalised
    rate inputs (qd, cube v, w) in training (build8's fix of the copycat shortcut of behaviour cloning)."""
    on = pol.norm(obs)
    if rate_noise > 0:
        on = on.clone()
        on[:, RATE_DIMS] = on[:, RATE_DIMS] + rate_noise * torch.randn_like(on[:, RATE_DIMS])
    a = pol.actor(on)
    d = codec.decode(a.float())
    e = ((d - rel).abs() / scale) * valid[..., None]
    return e.sum() / (valid.sum() * 6).clamp_min(1)


def train_bc(data: BCSet, out_dir, hidden=(512, 256, 128), steps=20000, batch=4096, lr=1e-3, wd=1e-5, seed=0,
             log=log_default, init_std=0.25, env_cfg=None, val_every=1000, rate_noise=0.0):
    """-> (policy, codec). Writes out_dir/{bc.pt, bc_curve.csv, codec.npz}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = data.dev
    codec = data.codec().to(dev)
    np.savez(out / "codec.npz", **codec.state())
    scale = data.scale().to(dev)
    pol = P.Policy(JE.OBS_DIM, JE.ACT_DIM, [n for _, n in JE.ACTION_GROUPS], hidden, hidden, "elu", init_std,
                   actor_out_gain=1.0).to(dev)
    tr = data.obs[data.train_rows].double()
    with torch.no_grad():
        pol.norm.mean.copy_(tr.mean(0))
        pol.norm.var.copy_(tr.var(0, unbiased=False))
        pol.norm.count.fill_(float(len(tr)))
    opt = torch.optim.AdamW(pol.actor.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device=dev).manual_seed(seed)
    f = open(out / "bc_curve.csv", "w")
    f.write("step,loss,val_loss,val_mae_arm_mrad,val_mae_jaw_mrad,lr\n")
    t0 = time.time()
    run = 0.0
    for s in range(1, steps + 1):
        frac = 0.5 * (1 + math.cos(math.pi * s / steps))
        for gr in opt.param_groups:
            gr["lr"] = lr * (0.05 + 0.95 * frac) * min(1.0, s / 300)
        i = data.train_rows[torch.randint(len(data.train_rows), (batch,), device=dev, generator=g)]
        loss = bc_loss(pol, codec, data.obs[i], data.rel[i], data.valid[i], scale, rate_noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(pol.actor.parameters(), 1.0)
        opt.step()
        run = 0.98 * run + 0.02 * float(loss) if s > 1 else float(loss)
        if s % val_every == 0 or s == steps:
            with torch.no_grad():
                vl, ea, ej, n = 0.0, 0.0, 0.0, 0
                for k in range(0, len(data.val_rows), 16384):
                    i = data.val_rows[k:k + 16384]
                    vl += float(bc_loss(pol, codec, data.obs[i], data.rel[i], data.valid[i], scale)) * len(i)
                    d = codec.decode(pol.actor(pol.norm(data.obs[i])).float())
                    e = (d - data.rel[i]).abs() * data.valid[i][..., None]
                    ea += float(e[..., :5].sum()) / 5
                    ej += float(e[..., 5].sum())
                    n += int(data.valid[i].sum())
            row = (s, run, vl / max(len(data.val_rows), 1), 1e3 * ea / max(n, 1), 1e3 * ej / max(n, 1),
                   opt.param_groups[0]["lr"])
            f.write(",".join(f"{x:.6g}" for x in row) + "\n")
            f.flush()
            log(f"  bc step {s}: loss {run:.4f} val {row[2]:.4f} mae arm {row[3]:.2f} jaw {row[4]:.2f} mrad "
                f"({time.time() - t0:.0f} s)")
    f.close()
    JE.save_bc(out / "bc.pt", pol, codec, env_cfg or {}, dict(steps=steps, batch=batch, lr=lr, hidden=list(hidden),
                                                             rate_noise=rate_noise))
    return pol, codec


# ---- 3. PPO fine-tune -----------------------------------------------------------------------------------------------
class JointPPO(P.PPO):
    """``ppo.PPO`` + critic warm-up, behaviour-cloning anchor and lenient statistics (module docstring)."""

    def __init__(self, env, policy, cfg, ref_actor=None, bc_coef=0.0, critic_warmup=0, device=None, freeze_norm=True):
        super().__init__(env, policy, cfg, device)
        if freeze_norm:          # keep the BC statistics: PPO's running update shifts the inputs of the BC actor
            policy.norm.update = lambda x: None
        self.ref = ref_actor
        self.bc_coef = float(bc_coef)
        self.critic_warmup = int(critic_warmup)
        self._lenient = 0.0

    @torch.no_grad()
    def collect(self):
        env = self.env
        step0 = env.step
        acc = torch.zeros((), dtype=torch.float64, device=self.dev)

        def step(a):
            out = step0(a)
            acc.add_((out[3]["episode"]["lenient"] & out[2]).sum())
            return out

        env.step = step
        try:
            st = super().collect()
        finally:
            env.step = step0
        self._last_len = float(acc)
        return st

    def update(self):
        al, pol, R = self.al, self.policy, self.roll
        warm = self.iteration < self.critic_warmup
        with torch.no_grad():
            returns, adv = P.gae(R.rewards, R.values, R.dones, self.last_values, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = R.T * R.B
        flat = lambda x: x.reshape(N, *x.shape[2:])                                  # noqa: E731
        obs, act, mask, old_lp, old_mu = flat(R.obs), flat(R.act), flat(R.mask).float(), flat(R.logp), flat(R.mu)
        old_v, ret, adv = flat(R.values), flat(returns), flat(adv)
        old_std_all = flat(R.std)
        clip = float(al["clip_param"])
        n_mb = int(al["num_mini_batches"])
        mb = N // n_mb
        out = dict(value_loss=0.0, surrogate_loss=0.0, entropy=0.0, kl=0.0, clip_frac=0.0, bc_loss=0.0)
        n_up = 0
        for _ in range(int(al["num_learning_epochs"])):
            perm = torch.randperm(N, generator=self.gen, device=self.dev)
            for k in range(n_mb):
                i = perm[k * mb:(k + 1) * mb]
                m = mask[i]
                mu = pol.actor(obs[i])
                v = pol.value(obs[i])
                std = pol.std_of(None)
                lp = P.log_prob(mu, std, act[i], m)
                with torch.no_grad():
                    kl = P.kl_div(old_mu[i], old_std_all[i], mu, std, m)
                    kl_mean = float(kl.mean())
                if not warm and al.get("desired_kl") is not None and al.get("schedule") == "adaptive":
                    if kl_mean > al["desired_kl"] * 2.0:
                        self.lr = max(1e-6, self.lr / 1.5)
                    elif 0.0 < kl_mean < al["desired_kl"] / 2.0:
                        self.lr = min(1e-3, self.lr * 1.5)
                self.opt.param_groups[0]["lr"] = 0.0 if warm else self.lr
                self.opt.param_groups[1]["lr"] = self.critic_lr
                ratio = torch.exp(lp - old_lp[i])
                a_ = adv[i]
                surr = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - clip, 1 + clip)).mean()
                if al.get("use_clipped_value_loss", True):
                    vc = old_v[i] + (v - old_v[i]).clamp(-clip, clip)
                    vloss = torch.max((v - ret[i]) ** 2, (vc - ret[i]) ** 2).mean()
                else:
                    vloss = ((v - ret[i]) ** 2).mean()
                ent = P.entropy(std, m).mean()
                loss = float(al["value_loss_coef"]) * vloss
                if not warm:
                    loss = loss + surr - float(al["entropy_coef"]) * ent
                    if self.ref is not None and self.bc_coef > 0:
                        with torch.no_grad():
                            mu_ref = self.ref(obs[i])
                        bcl = ((mu - mu_ref) ** 2).mean()
                        loss = loss + self.bc_coef * bcl
                        out["bc_loss"] += float(bcl.detach())
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), float(al["max_grad_norm"]))
                self.opt.step()
                self.clamp_std()
                with torch.no_grad():
                    out["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
                out["value_loss"] += float(vloss.detach())
                out["surrogate_loss"] += float(surr.detach())
                out["entropy"] += float(ent.detach())
                out["kl"] += kl_mean
                n_up += 1
        res = {k: v / max(n_up, 1) for k, v in out.items()}
        res["critic_warmup"] = float(warm)
        return res

    def iterate(self):
        row = super().iterate()
        n = row.get("episodes", 0.0)
        row["lenient"] = float(self._last_len) / n if n else math.nan
        return row


def ppo_train_cfg(over=None):
    cfg = P.train_cfg_default(1)
    cfg["algorithm"].update(learning_rate=1e-4, desired_kl=0.01, entropy_coef=0.0, num_mini_batches=8,
                            num_learning_epochs=5)
    cfg["policy"].update(actor_out_gain=1.0, action_bias=None)
    cfg.update(num_steps_per_env=32, critic_lr=3e-4, init_std=[0.25, 0.25], std_floor=[0.05, 0.05],
               std_cap=[0.5, 0.5], std_reset=[0.25, 0.25])
    return P.merge(cfg, over)


# ---- 4. fixed-task episodes -----------------------------------------------------------------------------------------
EP_OUT = ("end", "success", "success10", "lifted", "end_xy", "end_yaw", "ret", "length", "steps", "lenient",
          "lenient_t")


def make_env(world, B, codec, device="cuda", nrd_ckpt=None, workers=0, mj_python=None, horizon=400, members="split",
             lead_clamp=True, floor_guard=True, hold_cap_s=None, hold_obs_s=0.0, dyn=None):
    """Fixed-task env (evaluation, round collection). ``hold_cap_s`` / ``hold_obs_s``: ``JE.hold_cap_kw`` (None / 0 =
    the task defaults). ``dyn``: robot dynamics (``JE.RobotDynConfig`` or its dict); None = the process-wide
    evaluation setting of ``$NEDM_SO101_DYN`` (unset = off), False = off."""
    ck = [str(Path(c).expanduser()) for c in ([nrd_ckpt] if isinstance(nrd_ckpt, (str, Path)) else (nrd_ckpt or []))]
    dyn = JE.dyn_from_env() if dyn is None else JE.dyn_config(dyn)
    cfg = JE.JointEnvConfig(world=world, num_envs=B, device=device, ckpts=ck, members=members, auto_reset=False,
                            stagger=False, starts="", horizon=int(horizon), mj_workers=workers, mj_python=mj_python,
                            lead_clamp=lead_clamp, floor_guard=floor_guard, dyn=dyn,
                            **JE.hold_cap_kw(hold_cap_s, hold_obs_s))
    return JE.JointVecEnv(cfg, codec)


@torch.no_grad()
def run_tasks(env: JE.JointVecEnv, tasks, teacher: JE.JointTeacher | None, student=None, src=None, noise=None,
              noise_seed=0, record=False, record_states=False, full_horizon=False, log=None, student_hook=None):
    """One episode per world on ``tasks`` (env.B rows). ``src`` [B] int: 0 = JT drives, 1 = student drives (default:
    student if given, else JT). ``noise`` [B] float: Gaussian noise of that many PPO std on the JT's normalised action
    (JT-driven worlds; counter-based draws from the task seed, decision index and ``noise_seed``). ``student``: object
    with ``reset(B)`` and ``__call__(ws, goal_pose, k) -> absolute targets [B, 5, 6]`` for the executed part.
    ``record``: per boundary (live worlds) q, qd, cube pose, the JT's absolute targets of the decision ``lab`` [5, 6]
    (the label: defined at every state), the JT observation ``obs`` [47] (relabelling by another JT), who drove; per
    episode the outcomes. ``record_states``: per 20 ms step q and
    cube pose [B, S, ...] (videos). ``full_horizon``: keep stepping after the task ended (videos).
    -> dict of numpy arrays."""
    from nedm.so101_policy.vision.demos import start_state
    from nedm.so101_policy.vision.teacher import hash_normal
    B, dev = env.B, env.dev
    if len(tasks["seeds"]) != B:
        raise ValueError(f"{len(tasks['seeds'])} tasks for {B} worlds")
    g = lambda k: torch.as_tensor(tasks[k], dtype=torch.float64, device=dev)              # noqa: E731
    mj = dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"], cube_quat=tasks["start_cube_quat"]) \
        if env.cfg.world == "mujoco" else None
    seeds = torch.as_tensor(tasks["seeds"], dtype=torch.long, device=dev)
    obs = env.reset_tasks(torch.arange(B, device=dev), start_state(tasks, dev), g("goal_xy"), g("goal_yaw"), seeds, mj)
    goal_pose = torch.cat((g("goal_xy"), g("goal_yaw")[:, None]), -1)
    if src is None:
        src = np.full(B, 1 if student is not None else 0)
    src_t = torch.as_tensor(np.asarray(src), dtype=torch.long, device=dev)
    stud = src_t == 1
    if bool(stud.any()) and student is None:
        raise ValueError("student-driven worlds need a student")
    if student is not None:
        student.reset(B)
    nz = torch.zeros(B, dtype=torch.float64, device=dev) if noise is None else \
        torch.as_tensor(np.broadcast_to(np.asarray(noise, np.float64), (B,)).copy(), device=dev)
    hseed = seeds * 1_000_003 + 7919 * int(noise_seed) + 23
    std = teacher.std.to(dev, torch.float64) if teacher is not None else None
    R = {k: [] for k in ("q", "qd", "cube_pos", "cube_quat", "lab", "obs", "live")} if record else None
    SS = {k: [] for k in ("q", "cube", "u")} if record_states else None
    keep = None
    t0 = time.time()
    t_student = 0.0

    def hook(e, j):
        if SS is not None:
            SS["q"].append(e.ws.q.float().cpu())
            SS["cube"].append(torch.cat((e.ws.cube_pos, e.ws.cube_quat), -1).float().cpu())

    for k in range(env.max_episode_length + 1):
        live = env.task.tr.end == 0
        if not bool(live.any()) and not full_horizon:
            break
        ws = env.ws
        a = lab = None
        if teacher is not None:
            a = teacher.act(obs).to(dev, torch.float64)
            lab = env.codec.targets(a, ws.q.to(torch.float64))
            if bool((nz > 0).any()):
                eps = hash_normal(hseed, k, JE.ACT_DIM).to(a)
                a = a + (nz[:, None] * std[None]) * eps
        tg = None
        if bool(stud.any()):
            ts = time.time()
            tg = student(ws, goal_pose, k).to(dev, torch.float64)
            t_student += time.time() - ts
        if R is not None:
            for name, v in (("q", ws.q), ("qd", ws.qd), ("cube_pos", ws.cube_pos), ("cube_quat", ws.cube_quat)):
                R[name].append(v.float().cpu())
            R["lab"].append(lab.float().cpu() if lab is not None else torch.zeros(B, JE.STEPS, 6))
            R["obs"].append(obs.float().cpu())
            R["live"].append(live.cpu())
        obs, _, done, ex = env.step(a, targets=tg, use_targets=stud if tg is not None else None,
                                    step_hook=hook if SS is not None else None)
        ep = ex["episode"]
        if keep is None:
            keep = {n_: v.clone() for n_, v in ep.items()}
        else:
            for n_, v in ep.items():
                keep[n_] = torch.where(done, v, keep[n_])
        if log is not None and k % 20 == 19:
            log(f"  decision {k + 1}: ended {int((env.task.tr.end != 0).sum())}/{B}")
    out = {n_: v.cpu().numpy() for n_, v in keep.items()}
    out["seeds"] = np.asarray(tasks["seeds"])
    out["goal_xy"] = np.asarray(tasks["goal_xy"], np.float32)
    out["goal_yaw"] = np.asarray(tasks["goal_yaw"], np.float32)
    out["src"] = np.asarray(src, np.int8)
    out["noise"] = nz.cpu().numpy()
    out["wall_s"] = np.float64(time.time() - t0)
    out["student_s"] = np.float64(t_student)
    if env.cfg.world == "mujoco":
        out["mj_reason"] = env.world.end_reason.clone().numpy().astype(np.int16)
    if R is not None:
        rows = {k_: torch.stack(v, 1).numpy() for k_, v in R.items()}
        n_rows = out["length"].astype(np.int64)
        for b in range(B):
            if not rows["live"][b, : n_rows[b]].all():
                raise RuntimeError(f"episode {b}: live rows are not a prefix")
        rb = np.repeat(np.arange(B), n_rows)
        rt = np.concatenate([np.arange(n) for n in n_rows]) if B else np.zeros(0, np.int64)
        for k_ in ("q", "qd", "cube_pos", "cube_quat", "lab", "obs"):
            out[f"row_{k_}"] = rows[k_][rb, rt]
        out["row_ep"], out["row_t"] = rb.astype(np.int32), rt.astype(np.int16)
        out["ep_row_start"] = np.concatenate([[0], np.cumsum(n_rows)[:-1]]).astype(np.int64)
    if SS is not None:
        out["state_q"] = torch.stack(SS["q"], 1).numpy()
        out["state_cube"] = torch.stack(SS["cube"], 1).numpy()
    return out


def summarize(r):
    """Strict / lenient success with Wilson intervals, failure causes (non-lenient episodes), lifted, end errors."""
    from nedm.so101_policy.vision.tray import wilson
    s = np.asarray(r["success"]).astype(bool)
    le = np.asarray(r["lenient"]).astype(bool)
    n = len(s)
    causes = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[~le], return_counts=True))}
    out = dict(n=n, lenient=float(le.mean()), lenient_ci95=list(wilson(int(le.sum()), n)), strict=float(s.mean()),
               strict_ci95=list(wilson(int(s.sum()), n)), success10=float(np.mean(r["success10"])),
               lifted=float(np.mean(r["lifted"])), causes_not_lenient=causes)
    if s.any():
        out["end_xy_mm_p50_strict"] = float(np.median(np.asarray(r["end_xy"])[s]) * 1e3)
    return out
