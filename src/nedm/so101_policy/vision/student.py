"""The camera student: LeRobot's ACT (``lerobot.policies.act``, LeRobot 0.6.1) used as a library, plus this study's
normalisation, training loop and checkpoints.

Environment (luffy): conda env ``so101_vision`` = a clone of ``so101_render`` (torch 2.12.0+cu130 for the 5090's sm_120,
Madrona through ``env_luffy.sh``) with ``lerobot==0.6.1 --no-deps``, torchvision 0.27.1+cu130 (the build that matches
torch 2.12) and LeRobot's light import dependencies (draccus, einops, huggingface-hub, safetensors, gymnasium,
opencv-python-headless, termcolor, tqdm). LeRobot 0.6.1 pins torch < 2.12, torchvision < 0.27 and numpy < 2.3; the env
breaks these pins on purpose (torch 2.12.0, torchvision 0.27.1, numpy 2.5.4): only ACT's model code is used, and its
unit tests pass here (``tests/so101_policy/test_vision.py``). LeRobot's processor pipeline (normaliser, device moves)
is not used; normalisation is done here with the same MEAN_STD rule.

Model (ACT, ``ACTConfig`` defaults unless ``StudentConfig`` says otherwise): ResNet18 (ImageNet weights, frozen batch
norm) per camera -> 512-d tokens of the layer-4 feature map; transformer encoder (4 layers) over [latent, state token,
goal token, image tokens]; decoder (1 layer, ACT's effective depth) with ``chunk`` queries -> ``chunk`` x 6 decoder
actions; CVAE (4-layer encoder over [cls, state, actions], latent 32, KL weight 10) in training, latent 0 at
inference. Inputs: ``observation.state`` = proprio [30] (q + decoder observation), ``observation.environment_state``
= goal [4] (x, y, sin 4 psi, cos 4 psi), ``observation.images.<camera>`` [3, H, W] per camera (``dataset.CAMERAS``).
Output: decoder-v2 actions at 10 Hz (chunk ``chunk`` = 20 decisions = 2 s by default). Execution: ACT's temporal
ensembling (``ACTTemporalEnsembler``, coefficient ``ensemble_coeff`` = 0.01, one model call per decision).

Loss: L1 on the normalised action chunk (padded steps after the episode end masked) + 10 KL. Option ``trigger_bce``:
a[4] is a classifier (its head output is a logit, BCE on a[4] > 0; ``predict_chunk`` reports 2 sigmoid - 1, so the
decoder threshold 0 is p = 0.5) and L1 covers the other 5 dims: the +-1 trigger under L1 regresses to the median and
the students became too conservative to trigger (build4_vision.md sec. 5). Optimiser AdamW
(weight decay 1e-4), learning rate ``lr`` for the transformer and ``lr_backbone`` for the ResNet, linear warm-up over
``warmup`` steps then constant; bf16 autocast; gradient-norm clip 10 (LeRobot's default).
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
class StudentConfig:
    chunk: int = 20
    res: int = V.RES
    cameras: tuple = V.CAMERAS
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
    # training
    batch: int = 64
    render_batch: int = 512            # states rendered per renderer call (then split into minibatches)
    lr: float = 5e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    warmup: int = 500
    grad_clip: float = 10.0
    shift_px: int = 4                  # DrQ random shift (dataset.random_shift); 0 = off
    sustain_place: bool = False        # dataset.sustain_place_labels (PLACE trigger held through the final CARRY line)
    privileged_cube: bool = False      # ABLATION: the true cube state (dataset.cube_vec) joins the goal token
    trigger_bce: bool = False          # a[4] as a classifier: its head output is a logit trained with binary cross
    trigger_bce_weight: float = 1.0    #   entropy (target a[4] > 0), reported as 2 sigmoid - 1 (module docstring)
    amp: bool = True                   # bf16 autocast
    seed: int = 0

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(StudentConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown student config keys {bad}")
        if "cameras" in d:
            d["cameras"] = tuple(d["cameras"])
        return StudentConfig(**{k: v for k, v in d.items() if k in known})


def act_config(c: StudentConfig, device="cuda"):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    inp = {OBS_STATE: PolicyFeature(FeatureType.STATE, (V.STATE_DIM,)),
           OBS_ENV: PolicyFeature(FeatureType.ENV, (V.GOAL_DIM + (V.PRIV_DIM if c.privileged_cube else 0),))}
    for cam in c.cameras:
        inp[IMG + cam] = PolicyFeature(FeatureType.VISUAL, (3, c.res, c.res))
    out = {"action": PolicyFeature(FeatureType.ACTION, (V.ACTION_DIM,))}
    return ACTConfig(input_features=inp, output_features=out, chunk_size=c.chunk, n_action_steps=1,
                     dim_model=c.dim_model, n_heads=c.n_heads, dim_feedforward=c.dim_feedforward,
                     n_encoder_layers=c.n_encoder_layers, n_decoder_layers=c.n_decoder_layers, use_vae=c.use_vae,
                     latent_dim=c.latent_dim, n_vae_encoder_layers=c.n_vae_encoder_layers, kl_weight=c.kl_weight,
                     dropout=c.dropout, temporal_ensemble_coeff=c.ensemble_coeff,
                     pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if c.pretrained else None,
                     device=str(device))


class Student(torch.nn.Module):
    """ACTPolicy + normalisation buffers (``DemoSet.stats``). Images in: float [n, C, 3, H, W] (``dataset.to_input``)."""

    def __init__(self, cfg: StudentConfig, stats: dict, device="cuda", decoder_cfg=None):
        super().__init__()
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.cfg = cfg
        self.decoder_cfg = dict(decoder_cfg or {})
        self.policy = ACTPolicy(act_config(cfg, device))
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32).clone())
        if getattr(cfg, "trigger_bce", False):      # the head's a[4] output is a raw logit (no normalisation)
            self.action_mean[4], self.action_std[4] = 0.0, 1.0
        self.to(device)

    def _batch(self, images, state, goal, action=None, is_pad=None):
        b = {OBS_STATE: ((state - self.state_mean) / self.state_std).clamp(-10, 10),
             OBS_ENV: ((goal - self.goal_mean) / self.goal_std).clamp(-10, 10)}
        for i, cam in enumerate(self.cfg.cameras):
            b[IMG + cam] = images[:, i]
        if action is not None:
            b["action"] = (action - self.action_mean) / self.action_std
            b["action_is_pad"] = is_pad
        return b

    def loss(self, images, state, goal, action, is_pad):
        if not getattr(self.cfg, "trigger_bce", False):
            loss, info = self.policy.forward(self._batch(images, state, goal, action, is_pad))
            return loss, info
        # LeRobot ACTPolicy.forward's loss with a[4] as a classifier: L1 on the other 5 dims (padded steps masked),
        # BCE-with-logits on a[4] (target a[4] > 0), + kl_weight KL of the CVAE
        import torch.nn.functional as F
        from lerobot.utils.constants import OBS_IMAGES
        b = self._batch(images, state, goal, action, is_pad)
        b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
        hat, (mu, logs) = self.policy.model(b)
        valid = (~is_pad)[..., None].float()
        dims = [0, 1, 2, 3, 5]
        l1 = ((b["action"][..., dims] - hat[..., dims]).abs() * valid).sum() / (valid.sum() * len(dims)).clamp_min(1)
        bce = F.binary_cross_entropy_with_logits(hat[..., 4].float(), (action[..., 4] > 0).float(), reduction="none")
        bce = (bce * valid[..., 0]).sum() / valid.sum().clamp_min(1)
        loss = l1 + self.cfg.trigger_bce_weight * bce
        info = {"l1_loss": float(l1), "bce_loss": float(bce)}
        if self.cfg.use_vae and logs is not None:
            kld = (-0.5 * (1 + logs - mu.pow(2) - logs.exp())).sum(-1).mean()
            info["kld_loss"] = float(kld)
            loss = loss + kld * self.cfg.kl_weight
        return loss, info

    @torch.no_grad()
    def predict_chunk(self, images, state, goal, max_batch=128):
        """-> decoder actions [n, chunk, 6] (de-normalised; not clipped). Runs max_batch samples per model call
        (GPU memory on the shared 5090: a 512-sample call hit torch's out-of-memory path)."""
        out = []
        for s in range(0, images.shape[0], max_batch):
            e = s + max_batch
            out.append(self.policy.predict_action_chunk(self._batch(images[s:e], state[s:e], goal[s:e])).float())
        a = torch.cat(out) if len(out) > 1 else out[0]
        a = a * self.action_std + self.action_mean
        if getattr(self.cfg, "trigger_bce", False):      # logit -> 2 p - 1 (p 0.5 = the decoder's threshold 0)
            a = a.clone()
            a[..., 4] = 2 * torch.sigmoid(a[..., 4]) - 1
        return a

    @torch.no_grad()
    def val_l1(self, images, state, goal, action, is_pad):
        """Inference-path (latent 0) L1 of the de-normalised actions per dim, padded steps masked -> [6]."""
        pred = self.predict_chunk(images, state, goal)
        m = (~is_pad)[..., None].float()
        return ((pred - action).abs() * m).sum((0, 1)) / m.sum((0, 1)).clamp_min(1)

    def save(self, path, **extra):
        torch.save(dict(cfg=asdict(self.cfg), decoder_cfg=self.decoder_cfg, state_dict=self.state_dict(),
                        stats={k: getattr(self, k).cpu() for k in STAT_KEYS}, **extra), path)


STAT_KEYS = ("state_mean", "state_std", "goal_mean", "goal_std", "action_mean", "action_std")


def load_student(path, device="cuda"):
    ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    cfg = StudentConfig.from_dict(ck["cfg"])
    cfg_np = StudentConfig(**{**asdict(cfg), "pretrained": False})       # weights come from the checkpoint
    s = Student(cfg_np, ck["stats"], device, ck.get("decoder_cfg"))
    s.load_state_dict(ck["state_dict"])
    s.cfg = cfg
    s.eval()
    return s, ck


# ---- training -------------------------------------------------------------------------------------------------------
def param_groups(student: Student, c: StudentConfig):
    bb = [p for n, p in student.policy.named_parameters() if n.startswith("model.backbone") and p.requires_grad]
    rest = [p for n, p in student.policy.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]
    return [dict(params=rest, lr=c.lr), dict(params=bb, lr=c.lr_backbone)]


def train(data: "V.DemoSet", views, cfg: StudentConfig, out_dir, max_minutes=30.0, max_steps=0, val_n=512,
          val_every=500, log_every=50, ckpt_minutes=10.0, device="cuda", log=print, extra_meta=None, init_from=None):
    """Train a student on ``data`` with images from ``views`` (``dataset.Views`` or any object with the same
    ``render``). Writes out_dir/learning_curve.csv, ckpt/step_<k>.pt every ``ckpt_minutes``, last.pt. -> Student."""
    out = Path(out_dir)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if init_from:                       # fine-tune (DAgger): weights AND normalisation of the earlier student
        st, ck0 = load_student(init_from, device)
        st.cfg = cfg
        stats = {k: getattr(st, k).cpu() for k in STAT_KEYS}
        if st.decoder_cfg != dict(data.decoder_cfg):
            raise ValueError(f"init student decoder {st.decoder_cfg} != data decoder {data.decoder_cfg}")
    else:
        stats = data.stats()
        st = Student(cfg, stats, device, data.decoder_cfg)
    opt = torch.optim.AdamW(param_groups(st, cfg), weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    # fixed validation batch (rendered once)
    vgen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    vb = data.batch(data.sample(val_n, vgen, "val" if len(data.val_rows) else "train"))
    vimg = views.render(vb["q"], vb["cube_pos"], vb["cube_quat"], vb["goal_pose"])
    meta = dict(cfg=asdict(cfg), decoder_cfg=data.decoder_cfg, demo_meta=data.meta, n_train_eps=data.n_train_eps,
                init_from=str(init_from) if init_from else None, relabelled=getattr(data, "relabelled", 0),
                dagger_eps=getattr(data, "dagger_eps", 0),
                n_val_eps=data.n_val_eps, train_rows=len(data.train_rows), val_rows=len(data.val_rows),
                stats={k: v.tolist() for k, v in stats.items()}, **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "learning_curve.csv", "w", newline="")
    wr = None
    t0 = time.time()
    t_render = t_train = 0.0
    step, samples, last_ck = 0, 0, time.time()
    buf, pos = None, 0
    agg = dict(loss=0.0, l1=0.0, kld=0.0, n=0)
    st.train()
    while True:
        el = (time.time() - t0) / 60
        if (max_minutes and el >= max_minutes) or (max_steps and step >= max_steps):
            break
        if buf is None or pos + cfg.batch > cfg.render_batch:
            tr = time.time()
            b = data.batch(data.sample(cfg.render_batch, gen))
            b["rgb"] = views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"])
            torch.cuda.synchronize() if torch.device(device).type == "cuda" else None
            t_render += time.time() - tr
            buf, pos = b, 0
        tt = time.time()
        sl = slice(pos, pos + cfg.batch)
        pos += cfg.batch
        img = V.to_input(buf["rgb"][sl], cfg.shift_px, gen)
        warm = min(1.0, (step + 1) / max(1, cfg.warmup))
        for g, lr0 in zip(opt.param_groups, base):
            g["lr"] = lr0 * warm
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.amp):
            loss, info = st.loss(img, buf["state"][sl], buf["goal"][sl], buf["action"][sl], buf["action_is_pad"][sl])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), cfg.grad_clip)
        opt.step()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - tt
        step += 1
        samples += cfg.batch
        agg["loss"] += float(loss.detach())
        agg["l1"] += info["l1_loss"]
        agg["kld"] += info.get("kld_loss", 0.0)
        agg["n"] += 1
        do_val = step % val_every == 0
        if step % log_every == 0 or do_val:
            row = dict(step=step, samples=samples, wall_min=round((time.time() - t0) / 60, 3),
                       loss=agg["loss"] / agg["n"], l1=agg["l1"] / agg["n"], kld=agg["kld"] / agg["n"],
                       grad_norm=float(gn), lr=opt.param_groups[0]["lr"], render_s=round(t_render, 1),
                       train_s=round(t_train, 1), views_per_s=round(views.views / max(views.seconds, 1e-9)),
                       samples_per_s=round(samples / max(time.time() - t0, 1e-9), 1),
                       gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0)
            agg = dict(loss=0.0, l1=0.0, kld=0.0, n=0)
            if do_val:
                st.eval()
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.amp):
                    l1 = torch.cat([st.val_l1(V.to_input(vimg[s:s + 128]), vb["state"][s:s + 128], vb["goal"][s:s + 128],
                                              vb["action"][s:s + 128], vb["action_is_pad"][s:s + 128])[None]
                                    for s in range(0, val_n, 128)]).mean(0)
                st.train()
                row.update(val_l1=float(l1.mean()), **{f"val_l1_a{k}": float(l1[k]) for k in range(V.ACTION_DIM)})
            if wr is None:
                keys = list(row) + ([] if do_val else ["val_l1"] + [f"val_l1_a{k}" for k in range(V.ACTION_DIM)])
                wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            log(json.dumps(row))
        if time.time() - last_ck >= ckpt_minutes * 60:
            st.save(out / "ckpt" / f"step_{step:06d}.pt", step=step, samples=samples, wall_min=(time.time() - t0) / 60)
            last_ck = time.time()
    st.save(out / "last.pt", step=step, samples=samples, wall_min=(time.time() - t0) / 60)
    f.close()
    summ = dict(steps=step, samples=samples, wall_min=(time.time() - t0) / 60, render_s=t_render, train_s=t_train,
                views=views.views, views_per_s=views.views / max(views.seconds, 1e-9))
    (out / "train_summary.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))
    st.eval()
    return st

