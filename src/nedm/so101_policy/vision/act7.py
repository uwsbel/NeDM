"""ACT camera student for the PPO teacher's action space (build5_vision.md, student (b)): LeRobot's ACT (0.6.1, as in
``student``) predicting a chunk of the teacher's 7-dim PRE-TANH actions through the gated decoder.

Inputs: ``observation.state`` = the 57 robot-side observation entries (``teacher.ROBOT_KEYS``), ``observation.
environment_state`` = goal (x, y, sin 4 psi, cos 4 psi), ``observation.images.<camera>`` (top, wrist, 288 x 288; round
1's cameras). Targets: the teacher's labels of decisions k .. k + chunk - 1 of the same episode (the DAgger states that
were visited), clipped to +- ``target_clip``, mean / std normalised per column over the read rows. Loss: ACT's L1,
masked by the decoder's read mask of each target decision (``teacher.read_mask``) and the episode end, + KL (CVAE).

Execution (``Act7Act``): one model call per decision; temporal ensembling over the predictions made since the decoder
mode last changed (``rollout.ModeEnsembler``: round 1's fix), ``ensemble`` = "mode" (default) / "none" (the newest
chunk's first action); ``coeff`` overrides the ensembling coefficient (ACT's convention: weight exp(-coeff i), i = 0
the oldest prediction; < 0 favours the newest); ``trigger_latest``: a[4] from the newest chunk's first action (round
1's option; off by default here: the teacher's trigger is a continuous state function, not a one-decision impulse).
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nedm.so101_policy.vision import dataset as V

OBS_STATE, OBS_ENV, IMG = "observation.state", "observation.environment_state", "observation.images."


@dataclass
class Act7Config:
    chunk: int = 10
    res: int = V.RES
    cameras: tuple = V.CAMERAS
    act_dim: int = 7
    robot_dim: int = 57
    goal_dim: int = 4
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    dropout: float = 0.1
    pretrained: bool = True
    ensemble_coeff: float = 0.01
    batch: int = 64
    render_batch: int = 256
    lr: float = 5e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    warmup: int = 500
    grad_clip: float = 10.0
    target_clip: float = 3.0
    amp: bool = True
    seed: int = 0

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(Act7Config.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note", "notes", "kind"})
        if bad:
            raise ValueError(f"unknown ACT7 config keys {bad}")
        if "cameras" in d:
            d["cameras"] = tuple(d["cameras"])
        return Act7Config(**{k: v for k, v in d.items() if k in known})


def act_config(c: Act7Config, device="cuda"):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    inp = {OBS_STATE: PolicyFeature(FeatureType.STATE, (c.robot_dim,)),
           OBS_ENV: PolicyFeature(FeatureType.ENV, (c.goal_dim,))}
    for cam in c.cameras:
        inp[IMG + cam] = PolicyFeature(FeatureType.VISUAL, (3, c.res, c.res))
    out = {"action": PolicyFeature(FeatureType.ACTION, (c.act_dim,))}
    return ACTConfig(input_features=inp, output_features=out, chunk_size=c.chunk, n_action_steps=1,
                     dim_model=c.dim_model, n_heads=c.n_heads, dim_feedforward=c.dim_feedforward,
                     n_encoder_layers=c.n_encoder_layers, n_decoder_layers=c.n_decoder_layers, use_vae=c.use_vae,
                     latent_dim=c.latent_dim, n_vae_encoder_layers=c.n_vae_encoder_layers, kl_weight=c.kl_weight,
                     dropout=c.dropout, temporal_ensemble_coeff=c.ensemble_coeff,
                     pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if c.pretrained else None,
                     device=str(device))


STAT_KEYS = ("robot_mean", "robot_std", "goal_mean", "goal_std", "action_mean", "action_std")


def data_stats(data, floor=1e-2, action_floor=5e-2):
    """Normalisation over the training rows (actions: masked per column)."""
    r = data.train_rows
    x = data.robot[r]
    g = data.goal[data.ep_index[r]]
    lab, m = data.label[r], data.mask[r]
    n = m.sum(0).clamp_min(1)
    mu = (lab * m).sum(0) / n
    sd = (((lab - mu) ** 2 * m).sum(0) / n).sqrt().clamp_min(action_floor)
    return dict(robot_mean=x.mean(0).cpu(), robot_std=x.std(0).clamp_min(floor).cpu(), goal_mean=g.mean(0).cpu(),
                goal_std=g.std(0).clamp_min(floor).cpu(), action_mean=mu.cpu(), action_std=sd.cpu())


class Act7(torch.nn.Module):
    def __init__(self, cfg: Act7Config, stats: dict, device="cuda"):
        super().__init__()
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.cfg = cfg
        self.policy = ACTPolicy(act_config(cfg, device))
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32).clone())
        self.to(device)

    def _batch(self, images, robot, goal, action=None, is_pad=None):
        b = {OBS_STATE: ((robot - self.robot_mean) / self.robot_std).clamp(-10, 10),
             OBS_ENV: ((goal - self.goal_mean) / self.goal_std).clamp(-10, 10)}
        for i, cam in enumerate(self.cfg.cameras):
            b[IMG + cam] = images[:, i]
        if action is not None:
            b["action"] = (action - self.action_mean) / self.action_std
            b["action_is_pad"] = is_pad
        return b

    def loss(self, images, robot, goal, action, mask, is_pad):
        from lerobot.utils.constants import OBS_IMAGES
        b = self._batch(images, robot, goal, action, is_pad)
        b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
        hat, (mu, logs) = self.policy.model(b)
        l1 = ((b["action"] - hat.float()).abs() * mask).sum() / mask.sum().clamp_min(1)
        info = {"l1_loss": float(l1.detach())}
        loss = l1
        if self.cfg.use_vae and logs is not None:
            kld = (-0.5 * (1 + logs - mu.pow(2) - logs.exp())).sum(-1).mean()
            info["kld_loss"] = float(kld.detach())
            loss = loss + kld * self.cfg.kl_weight
        return loss, info

    @torch.no_grad()
    def predict_chunk(self, images, robot, goal, max_batch=128):
        out = []
        for s in range(0, images.shape[0], max_batch):
            e = s + max_batch
            out.append(self.policy.predict_action_chunk(self._batch(images[s:e], robot[s:e], goal[s:e])).float())
        a = torch.cat(out) if len(out) > 1 else out[0]
        return a * self.action_std + self.action_mean

    def save(self, path, **extra):
        torch.save(dict(kind="act7", cfg=asdict(self.cfg), state_dict=self.state_dict(),
                        stats={k: getattr(self, k).cpu() for k in STAT_KEYS}, **extra), path)


def load_act7(path, device="cuda"):
    ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    cfg = Act7Config.from_dict(ck["cfg"])
    s = Act7(Act7Config(**{**asdict(cfg), "pretrained": False}), ck["stats"], device)
    s.load_state_dict(ck["state_dict"])
    s.cfg = cfg
    s.eval()
    return s, ck


@torch.no_grad()
def val_l1(st, data, views, rows, bs=128):
    """Inference-path (latent 0) masked L1 of the de-normalised first action and of the whole chunk, per column."""
    A = st.cfg.act_dim
    s1, c1 = torch.zeros(A, device=data.dev), torch.zeros(A, device=data.dev)
    sc, cc = torch.zeros(A, device=data.dev), torch.zeros(A, device=data.dev)
    for s in range(0, len(rows), bs):
        b = data.batch(rows[s:s + bs])
        img = V.to_input(views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
        with torch.autocast(data.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
            p = st.predict_chunk(img, b["robot"], b["goal"])
        e = (p - b["label"]).abs() * b["mask"]
        s1 += e[:, 0].sum(0)
        c1 += b["mask"][:, 0].sum(0)
        sc += e.sum((0, 1))
        cc += b["mask"].sum((0, 1))
    out = {f"val_l1_first_a{j}": float(s1[j] / c1[j].clamp_min(1)) for j in range(A)}
    out["val_l1_first"] = float(s1.sum() / c1.sum().clamp_min(1))
    out["val_l1_chunk"] = float(sc.sum() / cc.sum().clamp_min(1))
    return out


def train(data, views, cfg: Act7Config, out_dir, max_steps=6000, init_from=None, log=print, val_n=1024, val_every=500,
          log_every=100, device="cuda", extra_meta=None):
    """Train / fine-tune an ACT7 student on a ``perstep.DaggerSet`` built with ``chunk`` = cfg.chunk."""
    if data.chunk != cfg.chunk:
        raise ValueError(f"data chunk {data.chunk} != config chunk {cfg.chunk}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype_dev = torch.device(device).type
    cuda = dtype_dev == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if init_from:
        st, _ = load_act7(init_from, device)
        st.cfg = cfg
    else:
        st = Act7(cfg, data_stats(data), device)
    st.train()
    bb = [p for n, p in st.policy.named_parameters() if n.startswith("model.backbone") and p.requires_grad]
    rest = [p for n, p in st.policy.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]
    opt = torch.optim.AdamW([dict(params=rest, lr=cfg.lr), dict(params=bb, lr=cfg.lr_backbone)],
                            weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    vgen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    vrows = data.sample(min(val_n, max(1, len(data.val_rows))), vgen, "val") if len(data.val_rows) else None
    meta = dict(cfg=asdict(cfg), decoder_cfg=data.decoder_cfg, data_meta=data.meta, n_train_eps=data.n_train_eps,
                n_val_eps=data.n_val_eps, train_rows=len(data.train_rows), val_rows=len(data.val_rows),
                init_from=str(init_from or ""), robot_idx=data.robot_idx.tolist(), **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "learning_curve.csv", "w", newline="")
    wr = None
    t0 = time.time()
    t_render = t_train = 0.0
    step, samples = 0, 0
    buf, pos = None, 0
    agg = dict(loss=0.0, l1=0.0, kld=0.0, n=0)
    while step < max_steps:
        if buf is None or pos + cfg.batch > cfg.render_batch:
            tr = time.time()
            b = data.batch(data.sample(cfg.render_batch, gen))
            b["rgb"] = views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"])
            sync()
            t_render += time.time() - tr
            buf, pos = b, 0
        tt = time.time()
        sl = slice(pos, pos + cfg.batch)
        pos += cfg.batch
        img = V.to_input(buf["rgb"][sl])
        warm = min(1.0, (step + 1) / max(1, cfg.warmup))
        for g, lr0 in zip(opt.param_groups, base):
            g["lr"] = lr0 * warm
        with torch.autocast(dtype_dev, dtype=torch.bfloat16, enabled=cfg.amp):
            loss, info = st.loss(img, buf["robot"][sl], buf["goal"][sl], buf["label"][sl], buf["mask"][sl],
                                 buf["is_pad"][sl])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), cfg.grad_clip)
        opt.step()
        sync()
        t_train += time.time() - tt
        step += 1
        samples += cfg.batch
        agg["loss"] += float(loss.detach())
        agg["l1"] += info["l1_loss"]
        agg["kld"] += info.get("kld_loss", 0.0)
        agg["n"] += 1
        do_val = vrows is not None and (step % val_every == 0 or step == max_steps)
        if step % log_every == 0 or do_val:
            row = dict(step=step, samples=samples, wall_min=round((time.time() - t0) / 60, 3),
                       loss=agg["loss"] / agg["n"], l1=agg["l1"] / agg["n"], kld=agg["kld"] / agg["n"],
                       grad_norm=float(gn), lr=opt.param_groups[0]["lr"], render_s=round(t_render, 1),
                       train_s=round(t_train, 1), samples_per_s=round(samples / max(time.time() - t0, 1e-9), 1),
                       gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0)
            agg = dict(loss=0.0, l1=0.0, kld=0.0, n=0)
            if do_val:
                st.eval()
                row.update(val_l1(st, data, views, vrows))
                st.train()
            if wr is None:
                vk = ["val_l1_first", "val_l1_chunk"] + [f"val_l1_first_a{j}" for j in range(cfg.act_dim)]
                wr = csv.DictWriter(f, fieldnames=[k for k in row if k not in vk] + vk, extrasaction="ignore")
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
    st.eval()
    st.save(out / "last.pt", step=step, samples=samples, wall_min=(time.time() - t0) / 60, decoder_cfg=data.decoder_cfg,
            robot_idx=data.robot_idx.tolist())
    f.close()
    summ = dict(steps=step, samples=samples, wall_min=(time.time() - t0) / 60, render_s=t_render, train_s=t_train,
                views=views.views, views_per_s=views.views / max(views.seconds, 1e-9))
    (out / "train_summary.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))
    return st


class Act7Act:
    """Student object for ``teacher.MixAct`` / ``teacher.evaluate`` (module docstring)."""

    def __init__(self, student: Act7, views, robot_idx, device="cuda", ensemble="mode", trigger_latest=False,
                 coeff=None):
        if ensemble not in ("mode", "none"):
            raise ValueError(f"ensemble {ensemble!r}: mode or none")
        self.coeff = None if coeff is None else float(coeff)      # override of the config's ensemble_coeff (ACT's
        # convention: weight exp(-coeff i), i = 0 the OLDEST prediction; coeff < 0 favours the newest)
        self.st, self.views, self.dev = student, views, torch.device(device)
        self.robot_idx = torch.as_tensor(robot_idx, dtype=torch.long)
        self.ensemble, self.trigger_latest = ensemble, bool(trigger_latest)
        self.ens = None
        self.t_render = self.t_model = 0.0
        self.calls = 0

    def reset(self, env):
        from nedm.so101_policy.vision.rollout import ModeEnsembler
        c = self.st.cfg
        coeff = c.ensemble_coeff if self.coeff is None else self.coeff
        self.ens = ModeEnsembler(coeff, c.chunk) if self.ensemble == "mode" else None
        self.goal_pose = torch.cat((env.task.goal_xy, env.task.goal_yaw[:, None]), -1).to(self.dev, torch.float32)
        self.goal = V.goal_vec(env.task.goal_xy.cpu(), env.task.goal_yaw.cpu()).to(self.dev)

    @torch.no_grad()
    def __call__(self, obs, env):
        ws = env.ws
        t0 = time.time()
        rgb = self.views.render(ws.q.to(self.dev, torch.float32), ws.cube_pos.to(self.dev, torch.float32),
                                ws.cube_quat.to(self.dev, torch.float32), self.goal_pose)
        img = V.to_input(rgb)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        robot = obs.to(self.dev, torch.float32)[:, self.robot_idx.to(self.dev)]
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.st.cfg.amp):
            chunk = self.st.predict_chunk(img, robot, self.goal).float()
        if self.ens is not None:
            a = self.ens.update(chunk, env.dec.mode.to(self.dev))
            if self.trigger_latest:
                a = a.clone()
                a[:, 4] = chunk[:, 0, 4]
        else:
            a = chunk[:, 0]
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.t_render += t1 - t0
        self.t_model += time.time() - t1
        self.calls += 1
        return a.to(env.dev, torch.float64)
