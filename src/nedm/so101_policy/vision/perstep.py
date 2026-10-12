"""Per-step camera student (build5_vision.md, student (a)): one decoder action per decision from the current images.

Inputs (``teacher`` module docstring): the cameras of ``dataset.Views`` (top + wrist, 288 x 288, the goal plate drawn;
round 1's choice), the robot-side observation entries (``teacher.ROBOT_KEYS``, 57: joints, rates, TCP / gripper axes,
goal x, y, sin / cos 4 psi, decoder macro phase and state), mean / std normalised over the training rows.

Network (``PerStepConfig``): per camera an image encoder -> spatial softmax keypoints (expected x, y of ``keypoints``
learned 1x1-conv heat maps; Levine et al. 2016 / robomimic) + a projected global average pool; encoder ``resnet18``
(ImageNet weights, frozen batch norm, the first ``stages`` residual stages: 3 = layer3, 256 channels, stride 16 ->
18 x 18 at 288 px) or ``small`` (4 conv layers with GroupNorm from scratch, stride 8). Then an MLP (``hidden``, ELU)
over [keypoints and pools of every camera, robot vector] -> the 7-dim PRE-TANH action (the PPO teacher's action space;
the env applies tanh).

Loss: masked mean squared error to the teacher's deterministic pre-tanh action, the targets clipped to
+- ``target_clip`` (tanh(3) = 0.995), per column weights ``dim_weight``, the mask = the columns the decoder reads at the
decision (``teacher.read_mask``); rows where the decoder reads nothing are not sampled.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import teacher as TT


@dataclass
class PerStepConfig:
    encoder: str = "resnet18"          # resnet18 | small
    stages: int = 3                    # resnet stages kept (1-4)
    keypoints: int = 32                # spatial-softmax keypoints per camera
    pool_dim: int = 64                 # projected global average pool per camera (0 = off)
    hidden: tuple = (512, 512)
    res: int = V.RES
    cameras: tuple = V.CAMERAS
    act_dim: int = 7
    robot_dim: int = 57
    batch: int = 128
    render_batch: int = 256
    lr: float = 3e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    warmup: int = 300
    cosine: bool = True                # cosine decay of the learning rate to 10 % over max_steps
    target_clip: float = 3.0
    dim_weight: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    amp: bool = True
    seed: int = 0

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(PerStepConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note", "notes", "kind"})
        if bad:
            raise ValueError(f"unknown per-step config keys {bad}")
        for k in ("cameras", "hidden", "dim_weight"):
            if k in d:
                d[k] = tuple(d[k])
        return PerStepConfig(**{k: v for k, v in d.items() if k in known})


class SpatialSoftmax(nn.Module):
    """[n, C, h, w] -> [n, 2 k]: expected image coordinates (x, y in [-1, 1]) of k learned heat maps."""

    def __init__(self, c_in, k):
        super().__init__()
        self.proj = nn.Conv2d(c_in, k, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.k = k

    def forward(self, x):
        n, _, h, w = x.shape
        f = self.proj(x).flatten(2).float() / self.log_temp.exp()                 # [n, k, h w]
        p = torch.softmax(f, -1)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        ex = (p * gx.reshape(1, 1, -1)).sum(-1)
        ey = (p * gy.reshape(1, 1, -1)).sum(-1)
        return torch.cat((ex, ey), -1)


def _frozen_bn_resnet18(pretrained):
    import torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=w, norm_layer=FrozenBatchNorm2d)


class Encoder(nn.Module):
    """One camera: trunk -> (spatial-softmax keypoints, projected average pool)."""

    def __init__(self, cfg: PerStepConfig, pretrained=True):
        super().__init__()
        if cfg.encoder == "resnet18":
            r = _frozen_bn_resnet18(pretrained)
            layers = [r.conv1, r.bn1, r.relu, r.maxpool] + [getattr(r, f"layer{i}") for i in range(1, cfg.stages + 1)]
            self.trunk = nn.Sequential(*layers)
            c = (64, 128, 256, 512)[cfg.stages - 1]
            self.is_backbone = True
        elif cfg.encoder == "small":
            g = lambda c: nn.GroupNorm(8, c)                       # noqa: E731
            self.trunk = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), g(32), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), g(64),
                                       nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), g(64), nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 1, 1), g(128), nn.ReLU())
            c = 128
            self.is_backbone = False
        else:
            raise ValueError(f"encoder {cfg.encoder!r}")
        self.ssm = SpatialSoftmax(c, cfg.keypoints)
        self.pool = nn.Linear(c, cfg.pool_dim) if cfg.pool_dim else None
        self.out_dim = 2 * cfg.keypoints + cfg.pool_dim

    def forward(self, x):
        f = self.trunk(x)
        out = [self.ssm(f)]
        if self.pool is not None:
            out.append(F.elu(self.pool(f.mean((2, 3)).float())))
        return torch.cat(out, -1)


class PerStepStudent(nn.Module):
    """Images float [n, C, 3, H, W] (``dataset.to_input``) + robot vector [n, 57] -> pre-tanh action [n, 7]."""

    def __init__(self, cfg: PerStepConfig, stats: dict, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.ModuleList([Encoder(cfg, pretrained) for _ in cfg.cameras])
        d = sum(e.out_dim for e in self.enc) + cfg.robot_dim
        layers, k = [], d
        for h in cfg.hidden:
            layers += [nn.Linear(k, h), nn.ELU()]
            k = h
        layers.append(nn.Linear(k, cfg.act_dim))
        self.head = nn.Sequential(*layers)
        for k_, v in stats.items():
            self.register_buffer(k_, torch.as_tensor(v, dtype=torch.float32).clone())
        self.t_render = self.t_model = 0.0

    def forward(self, images, robot):
        feats = [e(images[:, i]) for i, e in enumerate(self.enc)]
        r = ((robot - self.robot_mean) / self.robot_std).clamp(-10, 10)
        return self.head(torch.cat(feats + [r.to(feats[0].dtype)], -1).float())

    def save(self, path, **extra):
        torch.save(dict(kind="perstep", cfg=asdict(self.cfg), state_dict=self.state_dict(),
                        stats={k: getattr(self, k).cpu() for k in ("robot_mean", "robot_std")}, **extra), path)


def load_perstep(path, device="cuda"):
    ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    cfg = PerStepConfig.from_dict(ck["cfg"])
    st = PerStepStudent(cfg, ck["stats"], pretrained=False)
    st.load_state_dict(ck["state_dict"])
    return st.to(device).eval(), ck


# ---- data -------------------------------------------------------------------------------------------------------------
class DaggerSet:
    """DAgger / teacher files (``teacher.collect`` output) on one device. Every episode is used (labels are the
    teacher's actions at the visited states, failed episodes included). Validation: ``val_frac`` of the episodes of the
    FIRST file (seeded), so the validation set is the same in every DAgger round. ``chunk`` > 1: targets of decisions
    k .. k + chunk - 1 of the same episode (ACT), masked after the episode end and (``same_mode``) after the first
    decoder-mode change (the mode-aware ensembler drops those predictions at execution). Sampled rows: rows where the
    decoder reads a column (``sample_all``: every row)."""

    def __init__(self, paths, device="cuda", chunk=1, val_frac=0.05, seed=0, target_clip=3.0, robot_idx=None,
                 sample_all=False, same_mode=True):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        files = [TT.load(p) for p in paths]
        self.meta = [f["meta"] for f in files]
        decs = {json.dumps(m["teacher"]["decoder_cfg"], sort_keys=True, default=str) for m in self.meta}
        if len(decs) > 1:
            raise ValueError("DAgger files with different decoders")
        self.decoder_cfg = self.meta[0]["teacher"]["decoder_cfg"]
        keys = set.intersection(*[set(f) for f in files]) - {"meta"}
        d = TT.D.concat([{k: v for k, v in f.items() if k in keys} for f in files])
        n_ep = [len(f["ep_end"]) for f in files]
        E = len(d["ep_end"])
        rng = np.random.default_rng(int(seed))
        first = rng.permutation(n_ep[0])
        val_eps = np.sort(first[: int(round(val_frac * n_ep[0]))])
        is_val = np.zeros(E, bool)
        is_val[val_eps] = True
        self.n_val_eps, self.n_train_eps = int(is_val.sum()), int(E - is_val.sum())
        self.chunk, self.dev = int(chunk), torch.device(device)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.dev)   # noqa
        from nedm.so101_policy import task as T
        from nedm.so101_policy.vec_env import obs_layout
        obs_rates = bool(self.meta[0]["teacher"].get("obs_rates", True))
        self.layout = obs_layout(T.TaskConfig(obs_rates=obs_rates))
        self.robot_idx = TT.robot_index(self.layout) if robot_idx is None else torch.as_tensor(robot_idx)
        obs = t(d["obs"])
        self.robot = obs[:, self.robot_idx.to(self.dev)]
        self.q, self.cube_pos, self.cube_quat = t(d["q"]), t(d["cube_pos"]), t(d["cube_quat"])
        self.label = t(d["label"]).clamp(-float(target_clip), float(target_clip))
        self.mode = t(d["mode"], torch.long)
        self.mask = TT.read_mask(d["mode"], self.label.shape[1]).to(self.dev)
        self.src = t(d["src"], torch.long)
        self.ep_index = t(d["ep_index"], torch.long)
        self.ep_start = t(d["ep_start"], torch.long)
        self.ep_len = t(d["ep_length"], torch.long)
        self.goal_pose = torch.cat((t(d["ep_goal_xy"]), t(d["ep_goal_yaw"])[:, None]), -1)
        self.goal = V.goal_vec(d["ep_goal_xy"], d["ep_goal_yaw"]).to(self.dev)
        row_val = torch.as_tensor(is_val, device=self.dev)[self.ep_index]
        use = torch.ones_like(row_val) if sample_all else self.mask.sum(-1) > 0
        # mode segments: a new segment at every decoder-mode change and at every episode start
        md = np.asarray(d["mode"], np.int64)
        new_seg = np.ones(len(md), bool)
        new_seg[1:] = (md[1:] != md[:-1]) | (np.asarray(d["ep_index"])[1:] != np.asarray(d["ep_index"])[:-1])
        self.seg = t(np.cumsum(new_seg), torch.long)
        self.same_mode = bool(same_mode)
        self.train_rows = torch.nonzero(use & ~row_val).flatten()
        self.val_rows = torch.nonzero(use & row_val).flatten()
        self.n_rows = len(self.q)

    def stats(self, floor=1e-2):
        r = self.train_rows
        x = self.robot[r]
        return dict(robot_mean=x.mean(0).cpu(), robot_std=x.std(0).clamp_min(floor).cpu())

    def sample(self, n, gen=None, split="train"):
        rows = self.train_rows if split == "train" else self.val_rows
        return rows[torch.randint(len(rows), (int(n),), device=self.dev, generator=gen)]

    def batch(self, rows):
        e = self.ep_index[rows]
        out = dict(q=self.q[rows], cube_pos=self.cube_pos[rows], cube_quat=self.cube_quat[rows],
                   goal_pose=self.goal_pose[e], robot=self.robot[rows], goal=self.goal[e], mode=self.mode[rows])
        if self.chunk == 1:
            out.update(label=self.label[rows], mask=self.mask[rows])
            return out
        k = torch.arange(self.chunk, device=self.dev)
        end = self.ep_start[e] + self.ep_len[e]
        idx = rows[:, None] + k[None]
        pad = idx >= end[:, None]
        idx = torch.minimum(idx, (end - 1)[:, None])
        keep = ~pad
        if self.same_mode:
            keep = keep & (self.seg[idx] == self.seg[rows][:, None])
        out.update(label=self.label[idx], mask=self.mask[idx] * keep[..., None].float(), is_pad=pad)
        return out


# ---- training ---------------------------------------------------------------------------------------------------------
def masked_mse(pred, label, mask, w):
    """-> (loss, per-column sums [A], per-column counts [A])."""
    se = (pred - label) ** 2 * mask
    per = se.sum(0)
    cnt = mask.sum(0)
    loss = (per * w).sum() / cnt.sum().clamp_min(1)
    return loss, per.detach(), cnt.detach()


@torch.no_grad()
def val_metrics(st, data, views, rows, bs=256):
    """Masked MSE per column, sign agreement of the trigger a[4] (free / hold) and the abort a[6] on rows."""
    from nedm.so101_policy.decoder import M
    A = st.cfg.act_dim
    per, cnt = torch.zeros(A, device=data.dev), torch.zeros(A, device=data.dev)
    agree = {"free": [0, 0], "hold": [0, 0], "pre": [0, 0]}
    for s in range(0, len(rows), bs):
        b = data.batch(rows[s:s + bs])
        img = V.to_input(views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
        with torch.autocast(data.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
            p = st(img, b["robot"])
        _, pp, cc = masked_mse(p.float(), b["label"], b["mask"], torch.ones(A, device=data.dev))
        per += pp
        cnt += cc
        for name, sel, j in (("free", b["mode"] == M["free"], 4), ("hold", b["mode"] == M["hold"], 4),
                             ("pre", (b["mode"] >= M["g_brake"]) & (b["mode"] <= M["g_settle"]), A - 1)):
            if bool(sel.any()):
                agree[name][0] += int(((p[sel, j] > 0) == (b["label"][sel, j] > 0)).sum())
                agree[name][1] += int(sel.sum())
    out = {f"val_mse_a{j}": float(per[j] / cnt[j].clamp_min(1)) for j in range(A)}
    out["val_mse"] = float(per.sum() / cnt.sum().clamp_min(1))
    out.update({f"val_trig_agree_{k}": (v[0] / v[1] if v[1] else float("nan")) for k, v in agree.items()})
    return out


def train(data: DaggerSet, views, cfg: PerStepConfig, out_dir, max_steps=4000, init_from=None, log=print, val_n=2048,
          val_every=500, log_every=100, device="cuda", extra_meta=None):
    """Train (or fine-tune from ``init_from``: weights AND normalisation) a per-step student. Writes out_dir/{meta.json,
    learning_curve.csv, train_summary.json, last.pt}. -> student (eval mode)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype_dev = torch.device(device).type
    cuda = dtype_dev == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if init_from:
        st, _ = load_perstep(init_from, device)
        st.cfg = cfg
    else:
        st = PerStepStudent(cfg, data.stats(), pretrained=True).to(device)
    st.train()
    bb = [p for e in st.enc if e.is_backbone for p in e.trunk.parameters() if p.requires_grad]
    bb_ids = {id(p) for p in bb}
    rest = [p for p in st.parameters() if p.requires_grad and id(p) not in bb_ids]
    groups = [dict(params=rest, lr=cfg.lr)] + ([dict(params=bb, lr=cfg.lr_backbone)] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    w = torch.as_tensor(cfg.dim_weight, dtype=torch.float32, device=device)
    vgen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    vrows = data.sample(min(val_n, max(1, len(data.val_rows))), vgen, "val") if len(data.val_rows) else None
    meta = dict(cfg=asdict(cfg), decoder_cfg=data.decoder_cfg, files=[m.get("file", "") for m in data.meta],
                data_meta=data.meta, n_train_eps=data.n_train_eps, n_val_eps=data.n_val_eps,
                train_rows=len(data.train_rows), val_rows=len(data.val_rows), init_from=str(init_from or ""),
                robot_idx=data.robot_idx.tolist(), **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "learning_curve.csv", "w", newline="")
    wr = None
    t0 = time.time()
    t_render = t_train = 0.0
    step, samples = 0, 0
    buf, pos = None, 0
    agg_loss, agg_n = 0.0, 0
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
        frac = min(1.0, (step + 1) / max(1, cfg.warmup))
        if cfg.cosine and step >= cfg.warmup:
            u = (step - cfg.warmup) / max(1, max_steps - cfg.warmup)
            frac = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * u))
        for g, lr0 in zip(opt.param_groups, base):
            g["lr"] = lr0 * frac
        with torch.autocast(dtype_dev, dtype=torch.bfloat16, enabled=cfg.amp):
            p = st(img, buf["robot"][sl])
        loss, _, _ = masked_mse(p.float(), buf["label"][sl], buf["mask"][sl], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), 10.0)
        opt.step()
        sync()
        t_train += time.time() - tt
        step += 1
        samples += cfg.batch
        agg_loss += float(loss.detach())
        agg_n += 1
        do_val = vrows is not None and (step % val_every == 0 or step == max_steps)
        if step % log_every == 0 or do_val:
            row = dict(step=step, samples=samples, wall_min=round((time.time() - t0) / 60, 3), loss=agg_loss / agg_n,
                       grad_norm=float(gn), lr=opt.param_groups[0]["lr"], render_s=round(t_render, 1),
                       train_s=round(t_train, 1), samples_per_s=round(samples / max(time.time() - t0, 1e-9), 1),
                       gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0)
            agg_loss, agg_n = 0.0, 0
            if do_val:
                st.eval()
                row.update(val_metrics(st, data, views, vrows))
                st.train()
            if wr is None:
                vk = (["val_mse"] + [f"val_mse_a{j}" for j in range(cfg.act_dim)]
                      + [f"val_trig_agree_{k}" for k in ("free", "hold", "pre")])
                keys = [k for k in row if k not in vk] + vk
                wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
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


# ---- acting -----------------------------------------------------------------------------------------------------------
class PerStepAct:
    """Student object for ``teacher.MixAct`` / ``teacher.evaluate``: render the world's state, robot entries of the
    observation -> pre-tanh action [B, 7] on the env's device."""

    def __init__(self, student: PerStepStudent, views, robot_idx, device="cuda"):
        self.st, self.views, self.dev = student, views, torch.device(device)
        self.robot_idx = torch.as_tensor(robot_idx, dtype=torch.long)
        self.t_render = self.t_model = 0.0
        self.calls = 0

    def reset(self, env):
        self.goal_pose = torch.cat((env.task.goal_xy, env.task.goal_yaw[:, None]), -1).to(self.dev, torch.float32)

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
        out = []
        for s in range(0, img.shape[0], 256):
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.st.cfg.amp):
                out.append(self.st(img[s:s + 256], robot[s:s + 256]).float())
        a = torch.cat(out)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.t_render += t1 - t0
        self.t_model += time.time() - t1
        self.calls += 1
        return a.to(env.dev, torch.float64)
