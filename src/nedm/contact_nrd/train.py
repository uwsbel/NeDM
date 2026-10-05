"""Staged, validation-only training of the unified contact NRD (AMD compute nodes).

Stages (the pool recipe, made system-agnostic):
  1. core: each moving body's 10 ms steps without its own contact, from every
     1 ms phase (subsampled), balanced over quantile bins of the velocity
     change; data-fitted linear readout, then Adam phases.
  2. collision network: binary cross-entropy on pairs in contact, the same
     pair near its contact (negatives), and random pairs.
  3. contact network: windows with a contact plus windows near one, with the
     pair's switch forced on there (true change ~ 0) so the correction fades
     out at the edge of the contact region; ridge-fitted linear skip, Adam.
  4. rollout refinement of the contact network.
Selection uses validation rollouts only; the test split is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from nedm.contact_nrd.evaluate import free_metrics, load_data, window_labels
from nedm.contact_nrd.model import PAIR_FEATURES, STATE, ContactGraphNRD, load


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def dilate(mask, reach):
    """Dilate a [N, T, P] boolean mask along time by +-reach."""
    x = mask.permute(0, 2, 1).to(torch.float32)
    return F.max_pool1d(x, 2 * reach + 1, stride=1, padding=reach).permute(0, 2, 1) > 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("AMD compute nodes only")
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    dev = args.device
    data = load_data(args.data, dev, max_episodes=cfg.get("max_episodes"))
    system = data["system"]
    states, splits, lengths, stride = data["states"], data["splits"], data["lengths"], data["stride"]
    n_ep, n_rec, n_mov = states.shape[:3]
    pairs = system["pairs"]
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    slot = {k: n for n, k in enumerate(moving)}
    labels = window_labels(data["contacts"], stride)                          # [N, W, P]
    n_win = labels.shape[1]
    previous = torch.cat((torch.zeros_like(labels[:, :1]), window_labels(data["contacts"], 1)[:, : n_win - 1]), 1)
    after = previous & ~labels                                                  # force may lag the overlap by a step
    valid = torch.arange(n_win, device=dev)[None, :] < (lengths - stride)[:, None]
    train = valid & (splits == 0)[:, None]
    touched = labels | after
    involved = torch.zeros(n_ep, n_win, n_mov, dtype=torch.bool, device=dev)
    for p, (i, j) in enumerate(pairs):
        involved[..., slot[i]] |= touched[..., p]
        if j in slot:
            involved[..., slot[j]] |= touched[..., p]
    val = torch.nonzero(splits == 1).flatten()
    if args.smoke:
        val = val[:24]
    rng = torch.Generator(device=dev).manual_seed(cfg["seed"] + 1000)

    def sample(pool, n):
        return pool[torch.randint(len(pool), (n,), device=dev, generator=rng)]

    def padded_times(t, context):
        raw = t[:, None] + torch.arange(1 - context, 1, device=dev) * stride
        return torch.where(raw < 0, t[:, None] % stride, raw)

    # ---- normalisation (training data only) ---------------------------------
    sample_windows = torch.nonzero(train)
    sample_windows = sample_windows[torch.randperm(len(sample_windows), device=dev, generator=rng)[:200000]]
    s0 = states[sample_windows[:, 0], sample_windows[:, 1]]                     # [M, D, 9]
    s1 = states[sample_windows[:, 0], sample_windows[:, 1] + stride]
    free = ~involved[sample_windows[:, 0], sample_windows[:, 1]]               # [M, D]
    flat0, flatd = s0.reshape(-1, STATE), (s1 - s0).reshape(-1, STATE)
    keep = free.reshape(-1)
    floors = torch.tensor([0.01] * 3 + [0.01] * 3 + [0.1] * 3, dtype=torch.float64, device=dev)
    state_std = flat0.std(0)
    mask = ((state_std > 1e-9) | (flatd[keep].std(0) > 1e-12)).to(torch.float64)
    norm = dict(state_mean=flat0.mean(0).tolist(), state_std=torch.maximum(state_std, floors).tolist(),
                delta_scale=flatd[keep].std(0).clamp_min(1e-6).tolist(), mask=mask.tolist(),
                gate_mean=[0.0] * PAIR_FEATURES, gate_std=[1.0] * PAIR_FEATURES,
                in_mean=[0.0] * PAIR_FEATURES, in_std=[1.0] * PAIR_FEATURES, out_mean=[0.0] * STATE, out_std=[1.0] * STATE)
    mc = {**cfg["model"], "architecture": "contact_graph_nrd_v1", "dt_s": data["dt"]}
    pretrained = None
    if cfg.get("init_checkpoint") or cfg.get("core_checkpoint"):
        pretrained = torch.load(cfg.get("init_checkpoint") or cfg["core_checkpoint"], map_location=dev, weights_only=False)
        if pretrained["runtime"]["data_sha256"] != data["index"]["data_sha256"]:
            raise ValueError("checkpoint trained on different data")
        norm = pretrained["normalization"]
    model = ContactGraphNRD(mc, system, norm).to(dev)
    with torch.no_grad():
        if pretrained is None:
            feats, _, back, _, mm = model.all_pair_features(s0)
            pool_f = torch.cat((feats.reshape(-1, PAIR_FEATURES), back.reshape(-1, PAIR_FEATURES)))
            model.gate_mean.copy_(pool_f.mean(0))
            model.gate_std.copy_(pool_f.std(0).clamp_min(1e-3))
            norm["gate_mean"], norm["gate_std"] = model.gate_mean.tolist(), model.gate_std.tolist()
    if cfg.get("init_checkpoint"):
        model.load_state_dict(pretrained["model_state_dict"])
    elif cfg.get("core_checkpoint"):
        model.load_state_dict({k: v for k, v in pretrained["model_state_dict"].items() if k.startswith(("backbone.", "core_head."))}, strict=False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(config=cfg, model_config=mc, system=system["name"], training_episodes=int((splits == 0).sum()),
                   validation_episodes=len(val), test_used_for_selection=False, host=platform.node(),
                   job_id=os.environ["SLURM_JOB_ID"], torch=torch.__version__,
                   parameters=sum(p.numel() for p in model.parameters()), data_sha256=data["index"]["data_sha256"],
                   source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    atomic_json(output / "run_config.json", runtime)
    start, best = time.perf_counter(), float("inf")
    batch = 64 if args.smoke else cfg.get("batch", 1024)

    def channel_scale(values):
        return torch.tensor(values, dtype=torch.float64, device=dev)
    free_scale = channel_scale(cfg.get("free_scale", [1e-6] * 3 + [5e-6] * 3 + [4e-4, 4e-4, 1e-3]))
    response_scale = channel_scale(cfg.get("response_scale", [2e-5] * 3 + [2e-4] * 3 + [1.5e-2, 1.5e-2, 5e-2]))
    rollout_scale = channel_scale(cfg.get("rollout_scale", [1e-4] * 3 + [1e-3] * 3 + [0.08, 0.08, 0.2]))

    def save(path, stage, update, validation=None):
        torch.save(dict(model_config=mc, system=system, normalization=norm, model_state_dict=model.state_dict(),
                        runtime=runtime, stage=stage, update=update, validation=validation), path)

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_s=round(time.perf_counter() - start, 1), **extra)
        with (output / "train_log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def check(stage, update, loss):
        nonlocal best
        model.eval()
        report = free_metrics(model, data, val)
        if not math.isfinite(report["selection_score"]):
            report["selection_score"] = 1e9
        if report["selection_score"] < best or not (output / "best.pt").exists():
            best = report["selection_score"]
            save(output / "best.pt", stage, update, report)
        save(output / "last.pt", stage, update, report)
        event(stage, update=update, loss=float(loss), rmse_p95_mm=1000 * report["rmse_m"]["p95"],
              at_target_median_mm=1000 * report["at_target_m"]["median"], at_target_p95_mm=1000 * report["at_target_m"]["p95"],
              eligible_p95_mm=1000 * report["eligible_at_target_m"]["p95"], events_ok=report["event_ok_fraction"],
              score=report["selection_score"])
        model.train()

    core_params = list(model.backbone.parameters()) + list(model.core_head.parameters())

    # ---- 1. core ------------------------------------------------------------
    if pretrained is None or cfg.get("core_init"):
        phase_keep = (torch.arange(n_win, device=dev) % cfg.get("free_phase_step", 5) == 0)[None, :]
        triples = []
        for b in range(n_mov):
            w = torch.nonzero(train & phase_keep & ~involved[..., b])
            triples.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
        triples = torch.cat(triples)
        with torch.no_grad():
            dvel = torch.cat([(states[c[:, 0], c[:, 1] + stride, c[:, 2], 3:6] - states[c[:, 0], c[:, 1], c[:, 2], 3:6]).norm(dim=-1)
                              for c in triples.split(1 << 20)])
            moving_steps = (states[triples[:, 0], triples[:, 1], triples[:, 2], 3:].abs().amax(-1) > 1e-6) | (dvel > 1e-9)
            triples, dvel = triples[moving_steps], dvel[moving_steps]
            bins = cfg.get("core_bins", 4)
            edges = torch.quantile(dvel[torch.randperm(len(dvel), device=dev)[:1000000]].float(), torch.linspace(0, 1, bins + 1, device=dev)[1:-1]).double()
            which = torch.bucketize(dvel, edges)
            groups = [triples[which == k] for k in range(bins) if (which == k).any()]
        runtime["core_groups"] = [len(g) for g in groups]
        atomic_json(output / "run_config.json", runtime)
        up = model.up.double()

        def rotate_about_up(v, angle):
            """Rotate [M, ..., 3] vectors about the up axis (Rodrigues)."""
            c, s = angle.cos(), angle.sin()
            while c.dim() < v.dim():
                c, s = c[..., None], s[..., None]
            k = up.expand_as(v)
            return v * c + torch.cross(k, v, dim=-1) * s + k * (k * v).sum(-1, keepdim=True) * (1 - c)

        def core_batch(n, augment=True):
            tr = torch.cat([sample(g, max(1, n // len(groups))) for g in groups])
            rows = states[tr[:, 0, None], padded_times(tr[:, 1], model.context), tr[:, 2, None]]     # [M, K, 9]
            target = states[tr[:, 0], tr[:, 1] + stride, tr[:, 2]] - rows[:, -1]
            if augment and cfg.get("rotation_augment", False):
                angle = torch.rand(len(tr), device=dev, generator=rng, dtype=torch.float64) * 2 * math.pi
                rows = torch.cat([rotate_about_up(rows[..., k:k + 3], angle) for k in (0, 3, 6)], -1)
                target = torch.cat([rotate_about_up(target[..., k:k + 3], angle) for k in (0, 3, 6)], -1)
            return rows, target

        if not cfg.get("core_init"):
            hist, target = core_batch(128 if args.smoke else 32768, augment=False)
            with torch.no_grad():
                emb = model.backbone((hist - model.state_mean) / model.state_std)[..., -1, :]
                if model.anchor_rest:
                    rest = torch.cat((hist[..., :3], torch.zeros_like(hist[..., 3:])), -1)
                    emb = emb - model.backbone((rest - model.state_mean) / model.state_std)[..., -1, :]
                design = emb if model.anchor_rest else torch.cat((emb, torch.ones_like(emb[:, :1])), -1)
                beta = torch.linalg.lstsq(design.cpu(), (target / model.delta_scale).cpu(), rcond=cfg.get("readout_rcond", 1e-2),
                                          driver="gelsd").solution.to(dev)
                model.core_head.weight.copy_(beta[:emb.shape[1]].T)
                model.core_head.bias.copy_(beta[emb.shape[1]] if not model.anchor_rest else torch.zeros_like(model.core_head.bias))
            event("core_readout_fit", groups=runtime["core_groups"])
        loss = torch.zeros(())
        for phase, spec in enumerate(cfg.get("core_phases", [{"updates": 50000, "lr": 3e-4, "batch": 1024}])):
            opt = torch.optim.Adam(core_params, lr=spec["lr"])
            updates = 5 if args.smoke else spec["updates"]
            for update in range(1, updates + 1):
                for group in opt.param_groups:
                    group["lr"] = spec["lr"] * (0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * update / updates)))
                hist, target = core_batch(64 if args.smoke else spec["batch"])
                loss = ((model.body_delta(hist) - target * model.mask) / free_scale).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core_params, 10.0)
                opt.step()
                if update % 10000 == 0:
                    event(f"core_phase{phase}", update=update, loss=float(loss))
        save(output / "free_core.pt", "free_core", 0)
        with torch.no_grad():
            pv = []
            for b in range(n_mov):
                w = torch.nonzero(valid & (splits == 1)[:, None] & ~involved[..., b])
                w = w[torch.randperm(len(w), device=dev, generator=rng)[:100000]]
                pv.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
            pv = torch.cat(pv)
            hist = states[pv[:, 0, None], padded_times(pv[:, 1], model.context), pv[:, 2, None]]
            err = torch.cat([model.body_delta(h) for h in hist.split(16384)]) - (states[pv[:, 0], pv[:, 1] + stride, pv[:, 2]] - hist[:, -1]) * model.mask
            core_report = {name: {q: float(torch.quantile(err[:, cols].norm(dim=-1), qq)) for q, qq in (("median", 0.5), ("p95", 0.95))}
                           for name, cols in (("pos_m", [0, 1, 2]), ("vel_mps", [3, 4, 5]), ("spin_radps", [6, 7, 8]))}
        atomic_json(output / "core_validation.json", core_report)
        event("core_validation", **{k: v["p95"] for k, v in core_report.items()})
    for p in core_params:
        p.requires_grad_(False)

    contact_params = list(model.contact.parameters()) + list(model.contact_skip.parameters())
    if not cfg.get("init_checkpoint"):
        # ---- 2. collision network --------------------------------------------
        positives = torch.nonzero(labels & train[..., None])                     # (n, t, p)
        near = dilate(labels, cfg.get("near_negative_records", 50)) & ~labels & ~after & train[..., None]
        near_neg = torch.nonzero(near)
        any_pair = torch.nonzero(train)

        def collision_batch(n):
            pos, neg = sample(positives, n // 3), sample(near_neg, n // 3)
            rand = sample(any_pair, n - 2 * (n // 3))
            rand = torch.cat((rand, torch.randint(len(pairs), (len(rand), 1), device=dev, generator=rng)), -1)
            picks = torch.cat((pos, neg, rand))
            s = states[picks[:, 0], picks[:, 1]]
            target = labels[picks[:, 0], picks[:, 1], picks[:, 2]].to(torch.float64)
            return s, picks[:, 2], target

        def collision_loss(s, which, target):
            forward, _, backward, _, mm = model.all_pair_features(s)
            logits = model.collision_logits(forward, backward, mm)
            chosen = logits[torch.arange(len(s), device=dev), which]
            return F.binary_cross_entropy_with_logits(chosen, target)

        gate_params = list(model.collision.parameters())
        opt = torch.optim.Adam(gate_params, lr=cfg.get("gate_lr", 1e-3))
        count = 5 if args.smoke else cfg.get("gate_updates", 30000)
        for update in range(1, count + 1):
            for group in opt.param_groups:
                group["lr"] = cfg.get("gate_lr", 1e-3) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * update / count)))
            loss = collision_loss(*collision_batch(2 * batch))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if update % 10000 == 0:
                event("collision_adam", update=update, loss=float(loss))
        if cfg.get("gate_lbfgs", 0) and not args.smoke:
            fixed = collision_batch(cfg.get("gate_lbfgs_batch", 120000))
            opt = torch.optim.LBFGS(gate_params, lr=1.0, max_iter=15, history_size=30, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-10, tolerance_change=1e-14)

            def gate_closure():
                opt.zero_grad()
                value = collision_loss(*fixed)
                value.backward()
                return value
            for _ in range(cfg["gate_lbfgs"]):
                opt.step(gate_closure)
        with torch.no_grad():
            s, which, target = collision_batch(60000)
            forward, _, backward, _, mm = model.all_pair_features(s)
            on = model.collision_logits(forward, backward, mm)[torch.arange(len(s), device=dev), which] >= 0
            t = target > 0.5
            event("collision_fit", fp=int((on & ~t).sum()), fn=int((~on & t).sum()), positives=int(t.sum()))
        for p in gate_params:
            p.requires_grad_(False)

        # ---- 3. contact network ----------------------------------------------
        reach = cfg.get("response_near_records", 30)
        forced = (dilate(labels, reach) if reach else labels) & ~after
        forced = forced | labels
        pos_windows = torch.nonzero(train & labels.any(-1))
        near_windows = torch.nonzero(train & forced.any(-1) & ~labels.any(-1))

        def build(windows):
            parts = []
            for w in windows.split(32768):
                s, y = states[w[:, 0], w[:, 1]], states[w[:, 0], w[:, 1] + stride]
                with torch.no_grad():
                    hist = states[w[:, 0, None], padded_times(w[:, 1], model.context)]
                    core = model.core_delta(s, hist)
                g = forced[w[:, 0], w[:, 1]].to(torch.float64)
                inv = torch.zeros(len(w), n_mov, dtype=torch.float64, device=dev)
                for p, (i, j) in enumerate(pairs):
                    inv[:, slot[i]] = torch.maximum(inv[:, slot[i]], g[:, p])
                    if j in slot:
                        inv[:, slot[j]] = torch.maximum(inv[:, slot[j]], g[:, p])
                parts.append((s, y, core, g, inv))
            return tuple(torch.cat([part[k] for part in parts]) for k in range(5))

        cached_pos, cached_near = build(pos_windows), build(near_windows) if len(near_windows) else None
        event("contact_cache", positive_windows=len(pos_windows), near_windows=len(near_windows))
        with torch.no_grad():
            # Normalisation and ridge-fitted linear skip from single-contact windows.
            s, y, core, g, inv = cached_pos
            single = g.sum(-1) == 1
            s, y, core, g = s[single], y[single], core[single], g[single]
            forward, rot_f, backward, rot_b, mm = model.all_pair_features(s)
            which = g.argmax(-1)
            ar = torch.arange(len(s), device=dev)
            feats = [forward[ar, which]]
            first = torch.tensor([slot[i] for i, _ in pairs], device=dev)[which]
            targets = [(y - s - core)[ar, first]]
            frames = [rot_f[ar, which] if rot_f is not None else None]
            mm_list = mm.tolist()
            for k, p in enumerate(mm_list):
                rows = which == p
                if rows.any():
                    feats.append(backward[ar[rows], k])
                    j_slot = slot[pairs[p][1]]
                    targets.append((y - s - core)[ar[rows], j_slot])
                    frames.append(rot_b[ar[rows], k] if rot_b is not None else None)
            x = torch.cat(feats)
            tgt = torch.cat(targets)
            if model.frame == "pair":
                rot = torch.cat(frames)
                tgt = torch.cat([(tgt[:, k:k + 3].unsqueeze(-2) @ rot).squeeze(-2) for k in (0, 3, 6)], -1)
            model.in_mean.copy_(x.mean(0))
            model.in_std.copy_(torch.maximum(x.std(0), torch.maximum(0.1 * model.gate_std, torch.full_like(model.gate_std, 1e-3))))
            model.out_mean.copy_(tgt.mean(0))
            model.out_std.copy_(tgt.std(0).clamp_min(1e-4))
            f = (x - model.in_mean) / model.in_std
            design = torch.cat((f, torch.ones_like(f[:, :1])), -1).cpu()
            yy = ((tgt - model.out_mean) / model.out_std).cpu()
            gram = design.T @ design + cfg.get("skip_ridge", 1e-3) * len(design) * torch.eye(design.shape[1], dtype=design.dtype)
            beta = torch.linalg.solve(gram, design.T @ yy).to(dev)
            model.contact_skip.weight.copy_(beta[:-1].T)
            model.contact_skip.bias.copy_(beta[-1])
            for key in ("in_mean", "in_std", "out_mean", "out_std"):
                norm[key] = getattr(model, key).tolist()
        event("contact_linear_fit", transitions=len(x))

        def contact_loss(s, y, core, g, inv):
            logits, d1, d2, mm = model.contact_terms(s)
            pred = s + core + model.scatter(s, g, d1, d2, mm)
            err = ((pred - y) * model.mask / response_scale).square().mean(-1)
            return (err * inv).sum() / inv.sum().clamp_min(1)

        def cached_batch(n):
            k = int(n * cfg.get("response_near_fraction", 0.5)) if cached_near is not None else 0
            out = []
            for cache, m in ((cached_pos, n - k), (cached_near, k)):
                if m:
                    idx = torch.randint(len(cache[0]), (m,), device=dev, generator=rng)
                    out.append(tuple(t[idx] for t in cache))
            return tuple(torch.cat([o[j] for o in out]) for j in range(5))

        for p in contact_params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(contact_params, lr=cfg.get("contact_lr", 1e-3))
        count = 5 if args.smoke else cfg.get("contact_updates", 80000)
        floor_lr = cfg.get("contact_lr_floor", 1e-3)
        for update in range(1, count + 1):
            for group in opt.param_groups:
                group["lr"] = cfg.get("contact_lr", 1e-3) * (floor_lr + (1 - floor_lr) * 0.5 * (1 + math.cos(math.pi * update / count)))
            loss = contact_loss(*cached_batch(64 if args.smoke else cfg.get("contact_batch", 4096)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(contact_params, 10.0)
            opt.step()
            if update % cfg.get("evaluate_every", 10000) == 0 or update == count:
                check("contact_adam", update, loss.detach())
    else:
        check("init", 0, torch.zeros(()))

    # ---- 4. rollout refinement --------------------------------------------
    model, _ = load(output / "best.pt", dev)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    contact_params = list(model.contact.parameters()) + list(model.contact_skip.parameters())
    for p in contact_params:
        p.requires_grad_(True)
    count = 2 if args.smoke else cfg.get("rollout_updates", 300)
    horizon = 4 if args.smoke else round(cfg.get("rollout_horizon_s", 0.5) / model.dt)
    rb = 4 if args.smoke else cfg.get("rollout_batch", 64)
    contact_windows = torch.nonzero(train & labels.any(-1))
    any_window = torch.nonzero(train)
    if count:
        opt = torch.optim.Adam(contact_params, lr=cfg.get("rollout_lr", 3e-6))
        for update in range(1, count + 1):
            starts = sample(contact_windows, rb // 2).clone()
            starts[:, 1] = (starts[:, 1] - torch.randint(1, horizon, (len(starts),), device=dev, generator=rng) * stride).clamp_min(0)
            w = torch.cat((starts, sample(any_window, rb - len(starts))))
            current = states[w[:, 0], w[:, 1]]
            hist = states[w[:, 0, None], padded_times(w[:, 1], model.context)]
            total, den = current.sum() * 0.0, 0
            for step in range(horizon):
                j = w[:, 1] + (step + 1) * stride
                active = j < lengths[w[:, 0]]
                y = states[w[:, 0], j.clamp_max(n_rec - 1)]
                pred = model(current, hist)
                total = total + (((pred - y) * model.mask / rollout_scale).square().mean((-1, -2)) * active).sum()
                den += int(active.sum())
                current = torch.where(active[:, None, None], pred, current)
                hist = model.advance_history(hist, current)
            loss = total / max(1, den)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(contact_params, 1.0)
            if not torch.isfinite(gnorm):
                raise RuntimeError("nonfinite rollout gradient")
            opt.step()
            if update % (2 if args.smoke else cfg.get("rollout_evaluate_every", 100)) == 0 or update == count:
                check("rollout", update, loss.detach())
    model, metadata = load(output / "best.pt", dev)
    report = free_metrics(model, data, val)
    atomic_json(output / "validation.json", report)
    done = dict(complete=True, smoke=args.smoke, test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                checkpoint_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(), best_stage=metadata["stage"],
                best_update=metadata["update"], validation_score=report["selection_score"],
                at_target_p95_mm=1000 * report["at_target_m"]["p95"], rmse_p95_mm=1000 * report["rmse_m"]["p95"])
    atomic_json(output / "complete.json", done)
    event("complete", **done)


if __name__ == "__main__":
    main()
