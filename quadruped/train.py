#!/usr/bin/env python3
"""Train the NN-ROM on a corpus produced by collect.py.

The model maps a window of (state, action) tokens to the per-token DELTA of the state:
`target[t] = state[t+1] - state[t]`. A causal transformer, so every position in the window
is a valid one-step prediction and the loss covers all of them.

This is a fresh implementation, not a wrapper. What it reproduces from the previous study
it reproduces deliberately, and the several things that study got wrong are listed here so
they are not re-derived by accident.

WHAT THE PREVIOUS STUDY ESTABLISHED, AND THIS KEEPS

Selection is on multi-step rollout error, not one-step validation loss. On the dose
ladder, one-step val_loss ranked corpora with rho = -0.80 -- the wrong sign -- while 10 s
rollout error ranked them +0.90. The mechanism is that fine-tuning rolls the model on its
own output and differentiates through the result, so one-step loss optimises a quantity
the downstream use never touches. Confirmed causally: two surrogates identical but for
`checkpoint_metric`, fine-tuned and scored the same way, and the rollout-selected one won
on all three command channels.

But raw `rollout_sel` is a lottery. It moves 27-46% between adjacent epochs with no trend,
and taking its argmin once handed three of seven arms an EPOCH-1 checkpoint. So it is
taken as a trailing median over >= 3 epochs, with at least 32 rollout episodes, and no
checkpoint is eligible until the history is full -- otherwise epoch 1 sets the bar against
itself. A model that predicts very little motion scores errdist near 1.0 for free, which
is exactly what an untrained one does.

`val_loss` is computed on a fixed seeded RANDOM subset of the validation split, never a
prefix. The prefix was the bug: a fixed cap over an episode-ordered split took the first N
windows, which on the largest corpus was 0.64% of it and ONE of eight command families.
The composition is therefore printed every run, not the count -- the count was always
right and the composition was always wrong.

The training sampler draws WITH REPLACEMENT, `steps_per_epoch * batch_size` per epoch.
The gradient budget is then fixed and independent of corpus size, which is what makes a
data-quantity comparison a statement about data rather than about compute. Sizing the
epoch by `len(dataset)` silently destroys that.

Normalisation statistics come from the TRAIN split only, and ride in the checkpoint as
buffers so a consumer cannot apply the wrong ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# Channels that are angles and must be unwrapped per segment before differencing. A
# wrapped angle produces a 2*pi delta at the seam, which is not a dynamics event, and the
# loss is computed on NORMALISED targets -- so the one channel with a pathology inflates
# its own std and gets weighted least by the very loss that would expose it. The previous
# study measured that: unwrapping pitch dropped its target_std from 0.34640 to 0.00305, a
# 113x rise in its effective loss weight.
CIRCULAR_SUFFIXES = ("_rad",)
NON_CIRCULAR_RAD = ()          # names ending _rad that are NOT angles; none at present
MAX_UNWRAPPED_STEP_RAD = math.pi


def circular_fields(fields):
    return [f for f in fields
            if f.endswith(CIRCULAR_SUFFIXES) and f not in NON_CIRCULAR_RAD]


# --------------------------------------------------------------------------- data

def load_params():
    p = HERE / "params"
    return yaml.safe_load((p / "presets.yaml").read_text())


def segment_files(corpus_dir: Path):
    """Every segment, with the episode index parsed out of its name.

    A SEGMENT is the sequence unit, not an episode: collect.py splits an episode at each
    push and drops the push window, so two segments of one episode are not contiguous in
    time and a window spanning them would cross a temporal jump.
    """
    out = []
    for f in sorted((corpus_dir / "episodes").glob("*.csv")):
        stem = f.stem                      # <corpus>_<ep:04d>_s<k>
        try:
            ep = int(stem.split("_")[-2])
        except (ValueError, IndexError):
            raise SystemExit(f"cannot parse an episode index out of {f.name!r}")
        out.append((ep, f))
    if not out:
        raise SystemExit(f"no segments under {corpus_dir / 'episodes'}")
    return out


DERIVED = ("grav_body_x", "grav_body_y", "grav_body_z")


def add_derived(rows):
    """Fill in the channels the collector deliberately does not log.

    `grav_body_*` is the gravity direction in the body frame. It is DERIVED FROM THE
    QUATERNION rather than from logged roll/pitch, because roll/pitch do not reproduce
    Chrono's value -- the preset note in params/presets.yaml says so explicitly, and the
    fine-tuner reads these three channels to build the policy observation out of a
    predicted state, so getting them from the wrong source would be wrong everywhere at
    once and only visible downstream.

    Computed here, once, from the same `transforms.projected_gravity` the collector and
    the policy use, so there is one implementation rather than three.
    """
    from quadruped.params import transforms as T  # noqa: PLC0415
    if not rows or DERIVED[0] in rows[0]:
        return rows
    need = ("quat_e0", "quat_e1", "quat_e2", "quat_e3")
    if any(q not in rows[0] for q in need):
        raise SystemExit(
            f"cannot derive {DERIVED[0]}..: the segment has no quaternion columns "
            f"({', '.join(need)}). Either the schema changed or this is not a corpus "
            f"this trainer can read.")
    for r in rows:
        g = T.projected_gravity(*(float(r[q]) for q in need))
        for name, v in zip(DERIVED, g, strict=True):
            r[name] = v
    return rows


def read_segment(path, state_fields, action_fields, rollout_fields, extra_fields=()):
    with path.open() as fh:
        rows = add_derived(list(csv.DictReader(fh)))
    if len(rows) < 2:
        return None
    missing = [f for f in list(state_fields) + list(action_fields) + list(rollout_fields)
               + list(extra_fields) if f not in rows[0]]
    if missing:
        raise SystemExit(f"{path.name} is missing {len(missing)} column(s): "
                         f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")

    def col(names):
        a = np.empty((len(rows), len(names)), dtype=np.float64)
        for j, n in enumerate(names):
            for i, r in enumerate(rows):
                v = r[n]
                a[i, j] = float(v) if v not in ("", None) else np.nan
        return a

    s = col(state_fields)
    a = col(action_fields)
    ro = col(rollout_fields)
    # Columns a consumer needs beside the model's inputs -- the fine-tune's control phase,
    # recorded command and recorded policy output. Never fed to the model.
    ex = col(list(extra_fields))

    # A channel that is entirely absent is a silent four-dimensional hole in the input.
    # Fail with the field name still in hand rather than training on it.
    for j, n in enumerate(state_fields):
        if not np.isfinite(s[:, j]).any():
            raise SystemExit(f"{path.name}: state channel {n!r} is all non-finite")
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        raise SystemExit(f"{path.name}: non-finite values survived collection")

    # Unwrap angles per segment, BEFORE differencing.
    circ = [j for j, n in enumerate(state_fields) if n in set(circular_fields(state_fields))]
    for j in circ:
        s[:, j] = np.unwrap(s[:, j])
    for j in [k for k, n in enumerate(rollout_fields) if n.endswith("_rad")]:
        ro[:, j] = np.unwrap(ro[:, j])

    # Guard the channels we did NOT unwrap: a jump past pi in one control step is either
    # an angle nobody declared or a divergence validity missed.
    for j, n in enumerate(state_fields):
        if j in circ:
            continue
        d = np.abs(np.diff(s[:, j]))
        if n.endswith("_rad") and d.size and d.max() > MAX_UNWRAPPED_STEP_RAD:
            raise SystemExit(f"{path.name}: {n!r} jumps {d.max():.3f} rad in one step "
                             f"and was not unwrapped; declare it circular or fix the data")
    return s, a, ro, ex


class Corpus:
    """Segments, their windows, and the statistics of the train split."""

    def __init__(self, corpus_dir: Path, preset: str, seq_len: int, extra_fields=()):
        cfg = load_params()
        if preset not in cfg["presets"]:
            raise SystemExit(f"unknown preset {preset!r}; have "
                             f"{sorted(cfg['presets'])}")
        spec = cfg["presets"][preset]
        # The preset nests field groups; flatten and check the declared dimension, because
        # a preset that says 36 and delivers 34 has been wrong before.
        fields = []
        for grp in spec["fields"]:
            fields.extend(grp if isinstance(grp, list) else [grp])
        if len(fields) != int(spec["dim"]):
            raise SystemExit(f"preset {preset!r} declares dim {spec['dim']} but lists "
                             f"{len(fields)} fields")
        self.state_fields = fields
        self.action_fields = list(cfg["actions"]["fields"])
        self.rollout_fields = list(cfg["rollout"]["fields"])
        self.extra_fields = list(extra_fields)
        self.seq_len = seq_len

        manifest = json.loads((corpus_dir / "manifest.json").read_text())
        self.manifest = manifest
        val_eps = set(manifest.get("split", {}).get("val_episodes", []))
        if not val_eps:
            raise SystemExit(
                "the manifest records no validation episodes. Training would run, the "
                "loss would fall, and there would be nothing to select on -- which is a "
                "silent failure, so it is refused here.")

        self.train, self.val = [], []
        for ep, f in segment_files(corpus_dir):
            got = read_segment(f, self.state_fields, self.action_fields,
                               self.rollout_fields, self.extra_fields)
            if got is None:
                continue
            s, a, ro, ex = got
            rec = {"episode": ep, "name": f.stem, "state": s, "action": a, "rollout": ro,
                   "extra": ex, "target": np.diff(s, axis=0)}
            # One transition fewer than rows: the last state has no successor.
            rec["n"] = rec["target"].shape[0]
            (self.val if ep in val_eps else self.train).append(rec)

        if not self.train:
            raise SystemExit("no training segments")
        if not self.val:
            raise SystemExit("no validation segments")

        self.stats = self._train_stats()

    def _train_stats(self):
        """Per-channel mean and std over the TRAIN split only.

        The val split is never opened for statistics. Leakage here would be invisible:
        the loss would simply be a little lower and nothing would look wrong.
        """
        def moments(key, dim):
            n = 0
            s1 = np.zeros(dim)
            s2 = np.zeros(dim)
            for r in self.train:
                x = r[key]
                n += x.shape[0]
                s1 += x.sum(axis=0)
                s2 += (x * x).sum(axis=0)
            m = s1 / n
            var = np.maximum(s2 / n - m * m, 0.0)
            return m, np.maximum(np.sqrt(var), 1e-6)

        sm, ss = moments("state", len(self.state_fields))
        am, as_ = moments("action", len(self.action_fields))
        tm, ts = moments("target", len(self.state_fields))
        return {"state_mean": sm, "state_std": ss, "action_mean": am,
                "action_std": as_, "target_mean": tm, "target_std": ts}

    def windows(self, split):
        segs = self.train if split == "train" else self.val
        idx = []
        for si, r in enumerate(segs):
            # THE WINDOW RULE. A window never crosses a segment boundary.
            for k in range(max(r["n"] - self.seq_len + 1, 0)):
                idx.append((si, k))
        return segs, idx


# --------------------------------------------------------------------------- model

def build_model(torch, nn, state_dim, action_dim, cfg, stats):
    class Block(nn.Module):
        def __init__(self, d, h, p):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True,
                                              bias=False)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(),
                                     nn.Linear(4 * d, d, bias=False), nn.Dropout(p))

        def forward(self, x, mask):
            h = self.ln1(x)
            x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            return x + self.mlp(self.ln2(x))

    class NNROM(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg["n_embd"]
            self.inp = nn.Linear(state_dim + action_dim, d, bias=False)
            self.pos = nn.Embedding(cfg["block_size"], d)
            self.drop = nn.Dropout(cfg["dropout"])
            self.blocks = nn.ModuleList(
                [Block(d, cfg["n_head"], cfg["dropout"]) for _ in range(cfg["n_layer"])])
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, d, bias=False), nn.GELU(),
                                      nn.Linear(d, state_dim, bias=False))
            for name, t in [("state_mean", "state_mean"), ("state_std", "state_std"),
                            ("action_mean", "action_mean"), ("action_std", "action_std"),
                            ("target_mean", "target_mean"), ("target_std", "target_std")]:
                # Buffers, so the statistics travel in the checkpoint and a consumer
                # cannot pair the weights with the wrong normalisation.
                self.register_buffer(name, torch.tensor(stats[t], dtype=torch.float32))
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, state, action):
            s = (state - self.state_mean) / self.state_std
            a = (action - self.action_mean) / self.action_std
            x = self.inp(torch.cat([s, a], dim=-1))
            L = x.shape[1]
            x = self.drop(x + self.pos(torch.arange(L, device=x.device))[None])
            # Causal: token t may attend to <= t only. Without this the model sees the
            # future of its own window and every one-step number is meaningless.
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device),
                              diagonal=1)
            for b in self.blocks:
                x = b(x, mask)
            return self.head(self.ln_f(x))

        def predict_delta(self, state, action):
            """Denormalised next-state delta for every token."""
            return self.forward(state, action) * self.target_std + self.target_mean

    return NNROM()


# ----------------------------------------------------------------- rollout metric

def rollout_errdist(torch, model, corpus, episodes, horizon_s, dt_s, ctx,
                    dev=None):
    """Open-loop rollout error over distance travelled, at one horizon.

    ERRDIST, not raw position error: planar RMSE over the horizon, divided by the
    ground truth's MAXIMUM DISPLACEMENT from the branch start over that same horizon.

    The denominator is stated exactly because a normalised number hides what it divided
    by, and this project has already read one such number as a cross-system quantity when
    it was a within-system one, off by 25-52x. It is max displacement, not path length:
    a robot that drives in a circle returns near its start, so path length would flatter
    it. Both are defensible; they are not the same number and a table mixing them is
    comparing measurements rather than models.

    Dividing at all is what makes the metric comparable across domains and presets -- CRM
    episodes are shorter and slower than rigid ones, so a raw error favours whichever
    domain moves less. The scale to keep in mind: a model that predicts almost no motion
    scores about 1.0 for free, so 1.0 is the floor to beat, not the target.

    Pose is integrated OUTSIDE the propagated state, from the predicted body velocities
    and yaw rate, so the metric is the same physical quantity for every preset.
    """
    ix = {f: i for i, f in enumerate(corpus.state_fields)}
    need = ("vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps")
    for n in need:
        if n not in ix:
            raise SystemExit(f"rollout metric needs {n!r}, absent from this preset")
    n_steps = int(round(horizon_s / dt_s))
    out = []
    for r in episodes:
        if r["n"] < ctx + n_steps:
            continue
        # ON THE MODEL'S DEVICE. Built on the CPU by default, which crashed the first
        # real training run at the end of epoch 1 -- after 2000 steps had already run,
        # because the training loop moves its own batches and the evaluator did not.
        s = torch.tensor(r["state"][None, :ctx], dtype=torch.float32, device=dev)
        a_all = torch.tensor(r["action"][None], dtype=torch.float32, device=dev)
        x = y = th = 0.0
        gx, gy = r["rollout"][ctx - 1, 0], r["rollout"][ctx - 1, 1]
        dist = 0.0
        err2 = 0.0
        with torch.no_grad():
            for k in range(n_steps):
                t = ctx + k
                d = model.predict_delta(s, a_all[:, t - ctx:t])[0, -1]
                nxt = s[0, -1] + d
                s = torch.cat([s[:, 1:], nxt[None, None]], dim=1)
                vx, vy, wz = (float(nxt[ix[n]]) for n in need)
                th += wz * dt_s
                x += (vx * math.cos(th) - vy * math.sin(th)) * dt_s
                y += (vx * math.sin(th) + vy * math.cos(th)) * dt_s
                tx = r["rollout"][t, 0] - gx
                ty = r["rollout"][t, 1] - gy
                err2 += (x - tx) ** 2 + (y - ty) ** 2
                dist = max(dist, math.hypot(tx, ty))
        rmse = math.sqrt(err2 / n_steps)
        if dist > 1e-6:
            out.append(rmse / dist)
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def select_rollout_episodes(corpus, want):
    """Longest validation segments first, capped at `want`.

    Deterministic and stated: the metric is only comparable between runs if the episode
    set is. A rollout number taken from 12 episodes and one taken from 32 are different
    quantities, and putting them in one column compares measurements rather than models.
    """
    segs = sorted(corpus.val, key=lambda r: -r["n"])
    return segs[:want]


# --------------------------------------------------------------------------- train

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to data/<corpus>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="crm_baseline")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--dt-s", type=float, default=0.01)
    ap.add_argument("--max-val-windows", type=int, default=12800)
    ap.add_argument("--rollout-episodes", type=int, default=32)
    ap.add_argument("--rollout-horizon-s", type=float, default=10.0)
    ap.add_argument("--select-window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="exercise the code on a corpus too small to select on. Relaxes "
                         "the selection guards and STAMPS the checkpoint so it cannot be "
                         "mistaken for one that was selected properly.")
    a = ap.parse_args()

    if a.select_window < 3 and not a.smoke:
        raise SystemExit(
            "--select-window below 3 is refused. Raw-argmin rollout selection handed "
            "three of seven arms an epoch-1 checkpoint in the previous study, because a "
            "model that barely moves scores errdist near 1.0 for free.")
    # NOT A HARD COUNT ANY MORE, because the count was always a proxy for the thing that
    # actually matters. The previous study's rule -- at least 32 rollout episodes -- came
    # from observing that at 12 the metric moved 27-46% between adjacent epochs with no
    # trend. But the corpus decides how many long segments exist: a 60-episode pilot with
    # 20% long and 20% val yields two or three episodes that are BOTH, and reaching 32
    # that way needs roughly 800 episodes. A guard that cannot be satisfied gets bypassed.
    #
    # So the lottery is measured directly instead, at the end of training: median
    # adjacent-epoch movement of rollout_sel against its total range. The documented rule
    # is "if adjacent epochs move as far as the whole training does, the selection is a
    # lottery", and that is a property of the recorded metric, not of an episode count.
    # A run that fails it is STAMPED rather than silently trusted, and finetune.py refuses
    # a lottery-selected model the same way it refuses a smoke one.
    if a.rollout_episodes < 32 and not a.smoke:
        print(f"  NOTE: only {a.rollout_episodes} rollout episodes requested. The lottery "
              f"check at the end of this run is what decides whether the selection "
              f"stands; a low count usually fails it.")

    import torch
    from torch import nn

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                       else "cpu")

    corpus = Corpus(Path(a.corpus), a.preset, a.block_size)
    tr_segs, tr_idx = corpus.windows("train")
    va_segs, va_idx = corpus.windows("val")
    print(f"corpus {Path(a.corpus).name}  preset {a.preset} "
          f"({len(corpus.state_fields)}-D state, {len(corpus.action_fields)}-D action)")
    print(f"  train {len(tr_segs)} segments, {len(tr_idx):,} windows")
    print(f"  val   {len(va_segs)} segments, {len(va_idx):,} windows")

    # THE VALIDATION SUBSET IS RANDOM AND FIXED, never a prefix. Drawn once so it is the
    # same set every epoch, and reported by COMPOSITION rather than count -- the count was
    # always right in the failure this guards against, and the composition never was.
    rng = random.Random(a.seed + 1337)
    if len(va_idx) > a.max_val_windows:
        va_pick = rng.sample(range(len(va_idx)), a.max_val_windows)
    else:
        va_pick = list(range(len(va_idx)))
    eps_in = {va_segs[va_idx[i][0]]["episode"] for i in va_pick}
    print(f"  val subset {len(va_pick):,} windows "
          f"({100.0 * len(va_pick) / max(len(va_idx), 1):.2f}% of the split), "
          f"{len(eps_in)} of {len({r['episode'] for r in va_segs})} val episodes")

    cfg = {"block_size": a.block_size, "n_layer": a.n_layer, "n_head": a.n_head,
           "n_embd": a.n_embd, "dropout": a.dropout}
    model = build_model(torch, nn, len(corpus.state_fields), len(corpus.action_fields),
                        cfg, corpus.stats).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  model {nparam / 1e6:.2f} M parameters, {a.n_layer}L {a.n_head}H "
          f"{a.n_embd}d, context {a.block_size} ({a.block_size * a.dt_s:.2f} s)")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    total_steps = a.epochs * a.steps_per_epoch

    def lr_at(step):
        if step < a.warmup_steps:
            return a.lr * (step + 1) / a.warmup_steps
        p = (step - a.warmup_steps) / max(total_steps - a.warmup_steps, 1)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))

    def batch(segs, idx, picks):
        S = np.empty((len(picks), a.block_size, len(corpus.state_fields)), np.float32)
        A = np.empty((len(picks), a.block_size, len(corpus.action_fields)), np.float32)
        T = np.empty_like(S)
        for b, w in enumerate(picks):
            si, k = idx[w]
            r = segs[si]
            S[b] = r["state"][k:k + a.block_size]
            A[b] = r["action"][k:k + a.block_size]
            T[b] = r["target"][k:k + a.block_size]
        return (torch.from_numpy(S).to(dev), torch.from_numpy(A).to(dev),
                torch.from_numpy(T).to(dev))

    tstd = torch.tensor(corpus.stats["target_std"], dtype=torch.float32, device=dev)
    tmean = torch.tensor(corpus.stats["target_mean"], dtype=torch.float32, device=dev)

    def loss_of(pred_norm, target_raw):
        tgt = (target_raw - tmean) / tstd
        return torch.nn.functional.huber_loss(pred_norm, tgt, delta=a.huber_delta)

    metrics_path = out / "metrics.jsonl"
    history, best = [], float("inf")
    step = 0
    sampler = random.Random(a.seed)

    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run = 0.0
        for _ in range(a.steps_per_epoch):
            # WITH REPLACEMENT, a fixed number of draws per epoch. The gradient budget is
            # a property of the schedule, not of how much data happens to exist.
            picks = [sampler.randrange(len(tr_idx)) for _ in range(a.batch_size)]
            S, A, T = batch(tr_segs, tr_idx, picks)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = loss_of(model(S, A), T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            run += loss.detach().item()
            step += 1
        train_loss = run / a.steps_per_epoch

        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_pick), a.batch_size):
                picks = va_pick[i:i + a.batch_size]
                S, A, T = batch(va_segs, va_idx, picks)
                vl += loss_of(model(S, A), T).item()
                nb += 1
        val_loss = vl / max(nb, 1)

        ro_eps = select_rollout_episodes(corpus, a.rollout_episodes)
        rsel, ro_n = rollout_errdist(torch, model, corpus, ro_eps,
                                     a.rollout_horizon_s, a.dt_s, a.block_size, dev)
        history.append(rsel)

        # Trailing median, and nothing is eligible until the window is full.
        eligible = len(history) >= a.select_window
        smoothed = (float(np.median(history[-a.select_window:])) if eligible
                    else float("nan"))
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "rollout_sel": rsel, "rollout_episodes": ro_n,
               "rollout_sel_smoothed": smoothed, "lr": lr_at(step - 1),
               "seconds": round(time.perf_counter() - t0, 1)}
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

        ck = {"smoke": bool(a.smoke),
              "model": model.state_dict(), "optimizer": opt.state_dict(),
              "epoch": epoch, "config": vars(a),
              "state_fields": corpus.state_fields,
              "action_fields": corpus.action_fields,
              "rollout_fields": corpus.rollout_fields,
              "stats": {k: v.tolist() for k, v in corpus.stats.items()}}
        torch.save(ck, out / "last.pt")
        tag = ""
        if eligible and smoothed < best:
            best = smoothed
            torch.save(ck, out / "best.pt")
            tag = "  <- best"
        print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"rollout {rsel:.4f} (n={ro_n})  smoothed "
              f"{'--' if not eligible else f'{smoothed:.4f}'}{tag}", flush=True)

    if a.smoke:
        print("\nSMOKE RUN. The selection guards were relaxed, so this checkpoint was "
              "not selected on a usable rollout metric. It is stamped smoke=True and "
              "must not be scored or fine-tuned as if it were a real surrogate.")
    # THE LOTTERY CHECK. Lag-1 noise against total range, on the raw per-epoch metric.
    lottery, lag1, rng_ = None, float("nan"), float("nan")
    if len(history) >= 4:
        h = np.asarray(history, dtype=float)
        h = h[np.isfinite(h)]
        if h.size >= 4:
            lag1 = float(np.median(np.abs(np.diff(h))))
            rng_ = float(h.max() - h.min())
            ratio = lag1 / rng_ if rng_ > 0 else float("inf")
            lottery = bool(ratio > 0.5)
            print(f"\nselection stability: median adjacent-epoch move {lag1:.4f} against "
                  f"a total range of {rng_:.4f} ({100 * ratio:.0f}%)")
            if lottery:
                print("  LOTTERY. Adjacent epochs move about as far as the whole run "
                      "does, so which epoch won is close to arbitrary. best.pt is "
                      "stamped and downstream will refuse it; collect more long "
                      "held-out segments or raise --select-window.")
            else:
                print("  the selected epoch is distinguishable from its neighbours")
    # A SECOND CHECK, because the lottery test can pass on a useless run. This one did:
    # lag-1 noise against range came to 0.40, under the 0.5 threshold, while val_loss was
    # FLAT across 28 epochs -- 0.04443 to 0.04441 -- and the best epoch sat 1.2% below the
    # first. A metric that never improves has a small noise-to-range ratio simply because
    # its range IS noise, so "not a lottery" and "worth selecting on" are different
    # questions and both have to be asked.
    #
    # The failure that produces this is an undersized corpus: train loss fell 90x to a
    # 48x gap against val while val did not move, which is memorisation, not learning.
    no_trend = None
    if len(history) >= 10:
        h = np.asarray(history, dtype=float)
        h = h[np.isfinite(h)]
        if h.size >= 10:
            early = float(np.median(h[:max(3, h.size // 5)]))
            best = float(np.min(h))
            gain = (early - best) / early if early > 0 else 0.0
            no_trend = bool(gain < 0.15)
            print(f"selection trend: best rollout {best:.4f} against an early median of "
                  f"{early:.4f} ({100 * gain:.0f}% better)")
            if no_trend:
                print("  NO TREND. The metric never meaningfully improved, so the best "
                      "epoch is the luckiest one rather than the most trained. Usually "
                      "this means the corpus is too small -- check val_loss against "
                      "train_loss before adding epochs.")

    # ERRDIST ACROSS HORIZONS, recorded so the FLOOR can be checked where the model is
    # USED rather than where it is SELECTED. Those are different horizons and conflating
    # them was a real error: the 550-episode model scored 3.42 against the floor of 1.0 at
    # the 10 s selection horizon and was stamped worse-than-nothing, while at 0.30 s --
    # the length of an analytic fine-tuning branch -- it scored 0.451, less than half the
    # floor. It crosses 1.0 between 1 s and 2 s. The 10 s number is still the right one to
    # RANK checkpoints by (the previous study measured rho = +0.90 against transfer); it
    # is the wrong one to decide whether a model is usable for a 0.30 s rollout.
    horizon_profile = {}
    final_model = model
    if (out / "best.pt").exists():
        ck_b = torch.load(out / "best.pt", map_location=dev, weights_only=False)
        final_model.load_state_dict(ck_b["model"])
    final_model.eval()
    prof_eps = [r for r in sorted(corpus.val, key=lambda r: -r["n"])
                if r["n"] >= a.block_size + int(round(a.rollout_horizon_s / a.dt_s))
                ][:a.rollout_episodes]
    for hh in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, a.rollout_horizon_s):
        e_h, _n = rollout_errdist(torch, final_model, corpus, prof_eps, hh, a.dt_s,
                                  a.block_size, dev)
        horizon_profile[f"{hh:g}"] = e_h
    crossing = next((float(k) for k, v in sorted(horizon_profile.items(),
                                                 key=lambda kv: float(kv[0]))
                     if v >= 1.0), None)
    print("\nerrdist by horizon (selected model), against the predict-no-motion floor:")
    for k in sorted(horizon_profile, key=float):
        v = horizon_profile[k]
        print(f"  {float(k):5.1f} s  {v:7.3f}  {'better' if v < 1.0 else 'WORSE'}")
    print(f"  usable to roughly {crossing if crossing else 'the whole range'} s")

    # THE CHECK THAT SHOULD HAVE COME FIRST: is the model better than doing nothing?
    #
    # errdist divides trajectory error by distance travelled, so a model that predicts no
    # motion at all scores about 1.0. Anything at or above that is worse than the trivial
    # baseline, whatever its training curve looks like.
    #
    # This is here because the pilot run passed both of the cleverer guards and was still
    # useless. Lag-1 noise against range came to 10%, comfortably "not a lottery". The
    # rollout metric improved 33% from its early median, comfortably "has a trend". And
    # the best smoothed value was 2.163 -- more than twice as bad as predicting the robot
    # stands still. Two guards on the SHAPE of the curve, and neither asked what the
    # number meant.
    worse_than_nothing = None
    if len(history) >= a.select_window:
        h = np.asarray(history, dtype=float)
        h = h[np.isfinite(h)]
        if h.size >= a.select_window:
            sm = [float(np.median(h[max(0, i - a.select_window + 1):i + 1]))
                  for i in range(a.select_window - 1, h.size)]
            best_sm = min(sm) if sm else float("nan")
            # Judged at the USE horizon, 0.30 s, not at the selection horizon.
            use_e = horizon_profile.get("0.3", best_sm)
            worse_than_nothing = bool(use_e >= 1.0)
            print(f"selection floor: errdist {use_e:.4f} at the 0.30 s use horizon (the "
                  f"selection-horizon value is {best_sm:.4f}) against the floor of 1.0")
            if worse_than_nothing:
                print(f"  WORSE THAN NOTHING. This model is {best_sm:.1f}x further from "
                      f"the truth than a model that predicts the robot does not move. "
                      f"It is not a surrogate of anything and must not be fine-tuned in.")

    for f in ("last.pt", "best.pt"):
        pth = out / f
        if pth.exists():
            ck = torch.load(pth, map_location="cpu", weights_only=False)
            ck["selection_lottery"] = lottery
            ck["selection_no_trend"] = no_trend
            ck["worse_than_no_motion"] = worse_than_nothing
            ck["horizon_profile"] = horizon_profile
            ck["usable_to_s"] = crossing
            ck["selection_lag1"] = lag1
            ck["selection_range"] = rng_
            ck["rollout_episodes_used"] = ro_n
            torch.save(ck, pth)
    print(f"\nwrote {out}/last.pt and {out}/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
