"""Staged, validation-only training of the pool NRD (AMD compute nodes only).

Stages (as in the bouncing-ball Transformer v2 study):
  1. smooth core: data-fitted linear readout, then Adam and L-BFGS on
     contact-free 10 ms transitions (rolling, sliding, B at rest);
  2. contact switches: frozen core; balanced positives / near-contact
     negatives / free states; binary cross-entropy;
  3. bounce networks: data-fitted linear skip, then Adam and L-BFGS on contact
     transitions with the true labels switching the responses on;
  4. optional half-second rollout refinement of the bounce networks.
Every 1 ms phase of the 10 ms step is a training transition (time-invariant
dynamics), so each collision yields ~10 distinct transitions. Selection uses
validation rollouts from launch (phase 0) only; the test split is never read.
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

import numpy as np
import torch
from torch.nn import functional as F

from nedm.pool_ball.campaign import atomic_json
from nedm.pool_ball.evaluate import free_metrics, load_data, launch_gradient_check
from nedm.pool_ball.model import BALL, DIM, PoolNRD, load_pool

GROUPS = {"ab": [0], "a_cushion": [1, 2, 3, 4], "b_cushion": [5, 6, 7, 8]}


def window_labels(contacts, stride):
    """[N, T-stride, 3] labels (A-B, A cushion, B cushion) for 10 ms windows starting at each 1 ms."""
    c = contacts.to(torch.float32)
    grouped = torch.stack([c[..., idx].amax(-1) for idx in GROUPS.values()], -1)
    cum = torch.cat((torch.zeros_like(grouped[:, :1]), grouped.cumsum(1)), 1)
    return (cum[:, stride:] - cum[:, :-stride]) > 0.5


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
    data = load_data(args.data, args.device, max_episodes=cfg.get("max_episodes"))
    states, splits, lengths = data["states"], data["splits"], data["lengths"]
    stride = data["stride"]
    labels = window_labels(data["contacts"], stride)  # [N, T-stride, 3]
    if cfg["model"].get("contact") == "structured4":
        c = data["contacts"].to(torch.float32)
        cum = torch.cat((torch.zeros_like(c[:, :1]), c.cumsum(1)), 1)
        labels9 = (cum[:, stride:] - cum[:, :-stride]) > 0.5      # [N, T-stride, 9] in PAIRS order
    else:
        labels9 = None
    # Penalty force follows the overlap that labels it by one physics step, so
    # a window starting right after a contact can still carry force: keep such
    # windows out of the smooth (free) pools.
    previous = torch.cat((torch.zeros_like(labels[:, :1]), window_labels(data["contacts"], 1)[:, : labels.shape[1] - 1]), 1)
    after_contact = previous & ~labels
    n_windows = labels.shape[1]
    valid = torch.arange(n_windows, device=args.device)[None, :] < (lengths - stride)[:, None]
    train = valid & (splits == 0)[:, None]
    any_contact = labels.any(-1)
    free_pairs = torch.nonzero(train & ~any_contact & ~after_contact.any(-1))
    contact_pairs = torch.nonzero(train & any_contact)
    module_pairs = {k: torch.nonzero(train & labels[..., j]) for j, k in enumerate(GROUPS)}
    near = torch.zeros_like(train)
    margin = int(cfg.get("near_margin_steps", 5)) * stride
    for offset in range(-margin, margin + 1, max(1, stride // 2)):
        if offset:
            near[contact_pairs[:, 0], (contact_pairs[:, 1] + offset).clamp(0, n_windows - 1)] = True
    near_pairs = torch.nonzero(near & train & ~any_contact)
    val = torch.nonzero(splits == 1).flatten()
    if args.smoke:
        val = val[:32]

    def truth(pairs):
        return states[pairs[:, 0], pairs[:, 1]], states[pairs[:, 0], pairs[:, 1] + stride], labels[pairs[:, 0], pairs[:, 1]]

    reference = states[free_pairs[:, 0], free_pairs[:, 1]]
    free_delta = states[free_pairs[:, 0], free_pairs[:, 1] + stride] - reference
    sample_idx = torch.randperm(len(reference), device=args.device)[:200000]
    norm = dict(mean=reference[sample_idx].mean(0).cpu().tolist(),
                std=reference[sample_idx].std(0).clamp_min(0.01).cpu().tolist(),
                delta_mean=free_delta[sample_idx].mean(0).cpu().tolist(),
                delta_scale=free_delta[sample_idx].std(0).clamp_min(1e-4).cpu().tolist())
    # Per-ball statistics pooled over A and B, from moving balls only.
    pooled = torch.cat((reference[sample_idx][:, :BALL], reference[sample_idx][:, BALL:]))
    pooled_delta = torch.cat((free_delta[sample_idx][:, :BALL], free_delta[sample_idx][:, BALL:]))
    moving = pooled[:, 2:4].norm(dim=-1) > 1e-3
    norm.update(ball_mean=pooled[moving].mean(0).cpu().tolist(), ball_std=pooled[moving].std(0).clamp_min(0.01).cpu().tolist(),
                ball_delta_scale=pooled_delta[moving].std(0).clamp_min(1e-5).cpu().tolist())
    del reference, free_delta, pooled, pooled_delta
    pretrained = None
    core_init = cfg.get("core_init")
    refine_only = bool(cfg.get("init_checkpoint"))
    if refine_only:
        cfg["core_checkpoint"] = cfg["init_checkpoint"]
    if cfg.get("core_checkpoint") or core_init:
        cfg_core = cfg.get("core_checkpoint") or core_init
        # Reuse a trained, frozen Transformer core (and its normalisation).
        pretrained = torch.load(cfg_core, map_location=args.device, weights_only=False)
        if pretrained["runtime"]["data_sha256"] != data["index"]["model_data_sha256"]:
            raise ValueError("core checkpoint was trained on different data")
        norm = pretrained["normalization"]
        for key in ("core", "context", "layers", "embedding", "heads", "affine_preserving"):
            if pretrained["model_config"].get(key) != cfg["model"].get(key):
                raise ValueError(f"core checkpoint differs in {key}")
    mc = {**cfg["model"], "architecture": "pool_nrd_v1", "dt_s": data["dt"]}
    model = PoolNRD(mc, norm).to(args.device)
    if refine_only:
        # Continue a finished model: every module, normalisation included.
        model.load_state_dict(pretrained["model_state_dict"])
    elif pretrained is not None:
        state = {k: v for k, v in pretrained["model_state_dict"].items() if k.startswith(("backbone.", "core_head."))}
        missing = model.load_state_dict(state, strict=False)
        if any(k.startswith(("backbone.", "core_head.")) for k in missing.missing_keys):
            raise ValueError("core checkpoint is missing backbone weights")
    rng = torch.Generator(device=args.device).manual_seed(cfg["seed"] + 1000)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(config=cfg, model_config=mc, training_episodes=int((splits == 0).sum()), validation_episodes=len(val),
                   test_used_for_selection=False, host=platform.node(), job_id=os.environ["SLURM_JOB_ID"],
                   torch=torch.__version__, parameters=sum(p.numel() for p in model.parameters()),
                   data_sha256=data["index"]["model_data_sha256"], transitions=dict(free=len(free_pairs), contact=len(contact_pairs),
                   near=len(near_pairs), **{k: len(v) for k, v in module_pairs.items()}),
                   source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    atomic_json(output / "run_config.json", runtime)
    start, best = time.perf_counter(), float("inf")
    batch = 64 if args.smoke else cfg.get("batch", 512)
    def ball_scale(values):
        return torch.tensor(values, dtype=model.state_mean.dtype, device=args.device).repeat(2)
    # Per-channel tolerances per 10 ms step (rolling resistance changes v by
    # only 7e-4 m/s per step; a spin error matters like a velocity error
    # times ~88 through the rolling speed 5/7 (v + 0.4 R w)).
    free_scale = ball_scale(cfg.get("free_scale", [1e-6, 1e-6, 5e-6, 5e-6, 4e-4, 4e-4, 1e-3]))
    response_scale = ball_scale(cfg.get("response_scale", [2e-5, 2e-5, 2e-4, 2e-4, 1.5e-2, 1.5e-2, 5e-2]))
    rollout_scale = ball_scale(cfg.get("rollout_scale", [1e-4, 1e-4, 1e-3, 1e-3, 0.08, 0.08, 0.2]))

    def sample(pool, n):
        return pool[torch.randint(len(pool), (n,), device=args.device, generator=rng)]

    def padded_times(t):
        """History record indices ending at t, 10 ms apart; before the episode
        start, repeat the earliest record on the same 1 ms phase (as a rollout
        starting there repeats its first state)."""
        raw = t[:, None] + torch.arange(1 - model.context, 1, device=args.device) * stride
        return torch.where(raw < 0, t[:, None] % stride, raw)

    def history(pairs):
        return states[pairs[:, 0, None], padded_times(pairs[:, 1])]

    def save(path, stage, update, validation=None):
        torch.save(dict(model_config=mc, normalization=norm, model_state_dict=model.state_dict(), runtime=runtime,
                        stage=stage, update=update, validation=validation), path)

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
        event(stage, update=update, loss=float(loss), b_rmse_p95_mm=1000 * report["b_rmse_m"]["p95"],
              b_target_p95_mm=1000 * report["b_at_target_m"]["p95"], a_rmse_p95_mm=1000 * report["a_rmse_m"]["p95"],
              eligible_b_target_p95_mm=1000 * report["eligible_b_at_target_m"]["p95"], events_ok=report["event_ok_fraction"], score=report["selection_score"])
        model.train()

    # ---- 1. smooth core ------------------------------------------------
    core_params = list(model.backbone.parameters()) + list(model.core_head.parameters())
    radius = data["scene"]["radius_m"]
    loss = torch.zeros(())
    if pretrained is not None and not core_init:
        updates = 0
        event("reused_core", checkpoint=cfg["core_checkpoint"])
    elif model.core_mode == "per_ball":
        # Each ball's step is a smooth sample when that ball has no A-B or
        # cushion contact in the window (the other ball may be colliding).
        keep = (torch.arange(n_windows, device=args.device) % cfg.get("free_phase_step", 5) == 0)[None, :]
        triples = []
        touched = labels | after_contact
        for ball, involved in ((0, touched[..., 0] | touched[..., 1]), (1, touched[..., 0] | touched[..., 2])):
            pairs = torch.nonzero(train & keep & ~involved)
            triples.append(torch.cat((pairs, torch.full_like(pairs[:, :1], ball)), -1))
        triples = torch.cat(triples)

        def ball_state(tr, offset=0):
            rows = states[tr[:, 0], tr[:, 1] + offset]
            return torch.where(tr[:, 2:3] == 0, rows[:, :BALL], rows[:, BALL:])

        def slip(b):
            return torch.stack((b[:, 2] - radius * b[:, 5], b[:, 3] + radius * b[:, 4]), -1).norm(dim=-1)

        regimes = {}
        with torch.no_grad():
            parts = []
            for chunk in triples.split(1 << 20):
                b0, b1 = ball_state(chunk), ball_state(chunk, stride)
                moving = (b0[:, 2:4].norm(dim=-1) > 1e-4) | (b1[:, 2:4].norm(dim=-1) > 1e-4)
                s0, s1 = slip(b0) > 1e-3, slip(b1) > 1e-3
                code = torch.full_like(s0, 3, dtype=torch.int64)       # rolling
                code[s0 & s1] = 0                                      # sliding
                code[s0 != s1] = 1                                     # slide/roll change inside the step
                code[~moving] = 2                                      # at rest (anchored: exact zero)
                parts.append(code)
            codes = torch.cat(parts)
            for name, k in (("sliding", 0), ("transition", 1), ("rolling", 3)):
                regimes[name] = triples[codes == k]
        runtime["core_regimes"] = {k: len(v) for k, v in regimes.items()} | {"rest": int((codes == 2).sum())}
        atomic_json(output / "run_config.json", runtime)
        weights = cfg.get("regime_weights", {"sliding": 0.35, "transition": 0.2, "rolling": 0.45})

        def core_batch(n, augment=True):
            tr = torch.cat([sample(regimes[k], max(1, int(round(n * w)))) for k, w in weights.items() if len(regimes[k])])
            rows = states[tr[:, 0, None], padded_times(tr[:, 1])]
            hist = torch.where(tr[:, 2, None, None] == 0, rows[..., :BALL], rows[..., BALL:])
            target = ball_state(tr, stride) - hist[:, -1]
            if augment and cfg.get("rotation_augment", True):
                # The cloth is the same in every direction: rotate x-y
                # vectors (position, velocity, horizontal spin) about z.
                angle = torch.rand(len(tr), device=args.device, generator=rng, dtype=hist.dtype) * 2 * math.pi
                c, sn = angle.cos(), angle.sin()

                def rotate(v):
                    out = v.clone()
                    for i in (0, 2, 4):
                        x, y = v[..., i], v[..., i + 1]
                        cc, ss = (c, sn) if v.dim() == 2 else (c[:, None], sn[:, None])
                        out[..., i], out[..., i + 1] = cc * x - ss * y, ss * x + cc * y
                    return out
                hist, target = rotate(hist), rotate(target)
            return hist, target

        ball_free_scale = free_scale[:BALL]
        if model.affine_preserving and not core_init:
            hist, target = core_batch(128 if args.smoke else 32768, augment=False)
            with torch.no_grad():
                rest = torch.cat((hist[..., :2], torch.zeros_like(hist[..., 2:])), -1)
                design = model.ball_embedding(hist) - model.ball_embedding(rest)
                # Keep only the dominant (linear) directions: the block outputs
                # start tiny but nonzero, and fitting those weak directions
                # gives huge, fragile readout weights.
                beta = torch.linalg.lstsq(design.cpu(), (target / model.ball_delta_scale).cpu(), rcond=cfg.get("readout_rcond", 1e-2),
                                          driver="gelsd").solution.to(args.device)
                model.core_head.weight.copy_(beta.T)
                model.core_head.bias.zero_()
            event("data_fitted_core_readout", regimes=runtime["core_regimes"])
        opt = torch.optim.Adam(core_params, lr=cfg.get("core_lr", 3e-4))
        updates = 5 if args.smoke else cfg.get("core_warmup", 50000)
        for update in range(1, updates + 1):
            lr = cfg.get("core_lr", 3e-4) * (0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * update / updates)))
            for group in opt.param_groups:
                group["lr"] = lr
            hist, target = core_batch(batch)
            loss = ((model.ball_delta(hist) - target) / ball_free_scale).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core_params, 10.0)
            opt.step()
            if update % 5000 == 0:
                event("core_adam", update=update, loss=float(loss))
        if cfg.get("core_lbfgs", 0) and not args.smoke:
            hist, target = core_batch(cfg.get("lbfgs_batch", 32768))
            opt = torch.optim.LBFGS(core_params, lr=1.0, max_iter=15, history_size=30, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-12, tolerance_change=1e-15)

            def closure():
                opt.zero_grad()
                value = ((model.ball_delta(hist) - target) / ball_free_scale).square().mean()
                value.backward()
                return value
            for _ in range(cfg["core_lbfgs"]):
                opt.step(closure)
            loss = closure()
    else:
        if model.affine_preserving:
            pairs = sample(free_pairs, 128 if args.smoke else 16384)
            s, y, _ = truth(pairs)
            with torch.no_grad():
                h = model.backbone((history(pairs) - model.state_mean) / model.state_std)[:, -1]
                design = torch.cat((h, torch.ones_like(h[:, :1])), -1)
                target = (y - s - model.delta_mean) / model.delta_scale
                beta = torch.linalg.lstsq(design.cpu(), target.cpu(), rcond=1e-6, driver="gelsd").solution.to(args.device)
                model.core_head.weight.copy_(beta[:-1].T)
                model.core_head.bias.copy_(beta[-1])
            event("data_fitted_core_readout")
        opt = torch.optim.Adam(core_params, lr=cfg.get("core_lr", 3e-4))
        updates = 5 if args.smoke else cfg.get("core_warmup", 50000)
        for update in range(1, updates + 1):
            lr = cfg.get("core_lr", 3e-4) * (0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * update / updates)))
            for group in opt.param_groups:
                group["lr"] = lr
            pairs = sample(free_pairs, batch)
            s, y, _ = truth(pairs)
            loss = ((s + model.core_delta(s, history(pairs)) - y) / free_scale).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core_params, 10.0)
            opt.step()
            if update % 5000 == 0:
                event("core_adam", update=update, loss=float(loss))
        if cfg.get("core_lbfgs", 0) and not args.smoke:
            pairs = sample(free_pairs, cfg.get("lbfgs_batch", 32768))
            s, y, _ = truth(pairs)
            h = history(pairs)
            opt = torch.optim.LBFGS(core_params, lr=1.0, max_iter=15, history_size=30, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-12, tolerance_change=1e-15)

            def closure():
                opt.zero_grad()
                value = ((s + model.core_delta(s, h) - y) / free_scale).square().mean()
                value.backward()
                return value
            for _ in range(cfg["core_lbfgs"]):
                opt.step(closure)
            loss = closure()
    event("free_core", updates=updates, loss=float(loss.detach()))
    save(output / "free_core.pt", "free_core", updates)
    with torch.no_grad():
        pairs = torch.nonzero(valid & (splits == 1)[:, None] & ~any_contact)
        pairs = pairs[torch.randperm(len(pairs), device=args.device)[:200000]]
        errors = torch.cat([s + model.core_delta(s, history(p)) - y for p in pairs.split(8192) for s, y, _ in [truth(p)]])
        core_report = {name: float(torch.quantile(errors[:, cols].norm(dim=-1).double(), 0.95))
                       for name, cols in (("A_pos_m", [0, 1]), ("A_vel_mps", [2, 3]), ("A_spin_radps", [4, 5, 6]),
                                          ("B_pos_m", [7, 8]), ("B_vel_mps", [9, 10]), ("B_spin_radps", [11, 12, 13]))}
    if model.core_mode == "per_ball":
        # Per-regime one-step errors of the core on validation steps.
        with torch.no_grad():
            val_mask = valid & (splits == 1)[:, None]
            keep_v = (torch.arange(n_windows, device=args.device) % 5 == 0)[None, :]
            tri = []
            for ball, involved in ((0, labels[..., 0] | labels[..., 1] | after_contact[..., 0] | after_contact[..., 1]),
                                   (1, labels[..., 0] | labels[..., 2] | after_contact[..., 0] | after_contact[..., 2])):
                pv = torch.nonzero(val_mask & keep_v & ~involved)
                tri.append(torch.cat((pv, torch.full_like(pv[:, :1], ball)), -1))
            tri = torch.cat(tri)
            tri = tri[torch.randperm(len(tri), device=args.device)[:400000]]
            rows0 = states[tri[:, 0], tri[:, 1]]
            rows1 = states[tri[:, 0], tri[:, 1] + stride]
            b0 = torch.where(tri[:, 2:3] == 0, rows0[:, :BALL], rows0[:, BALL:])
            b1 = torch.where(tri[:, 2:3] == 0, rows1[:, :BALL], rows1[:, BALL:])
            hrows = states[tri[:, 0, None], padded_times(tri[:, 1])]
            hb = torch.where(tri[:, 2, None, None] == 0, hrows[..., :BALL], hrows[..., BALL:])
            err = torch.cat([model.ball_delta(h) for h in hb.split(16384)]) - (b1 - b0)
            def slip_of(b):
                return torch.stack((b[:, 2] - radius * b[:, 5], b[:, 3] + radius * b[:, 4]), -1).norm(dim=-1)
            s0, s1 = slip_of(b0) > 1e-3, slip_of(b1) > 1e-3
            moving = b0[:, 2:4].norm(dim=-1) > 1e-4
            for name, mask in (("sliding", s0 & s1 & moving), ("transition", (s0 != s1) & moving), ("rolling", ~s0 & ~s1 & moving)):
                if mask.any():
                    e = err[mask]
                    core_report[f"{name}_count"] = int(mask.sum())
                    for q in (0.5, 0.95):
                        core_report[f"{name}_vel_mps_q{int(q*100)}"] = float(torch.quantile(e[:, 2:4].norm(dim=-1).double()[:200000], q))
                        core_report[f"{name}_spin_radps_q{int(q*100)}"] = float(torch.quantile(e[:, 4:7].norm(dim=-1).double()[:200000], q))
                    core_report[f"{name}_vel_bias_along_mps"] = float((e[:, 2:4] * (b0[mask][:, 2:4] / b0[mask][:, 2:4].norm(dim=-1, keepdim=True))).sum(-1).mean())
    atomic_json(output / "free_core_validation_p95.json", core_report)
    event("free_core_validation_p95", **core_report)
    for p in core_params:
        p.requires_grad_(not cfg.get("freeze_core", True))

    if refine_only:
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = [p for m in model.modules_.values() for p in list(m.bounce.parameters()) + list(m.skip.parameters())]
        check("init", 0, torch.zeros(()))
    elif model.contact != "none":
        # ---- 2. contact switches ---------------------------------------
        if model.contact == "scalar_shared":
            gate_sets = {"any": (DIM, None)}
        else:
            gate_sets = {"ab": (DIM, None), "cushion": (BALL, None)}
        with torch.no_grad():
            for name, module in model.modules_.items():
                window_states = states[:, :n_windows][train]
                if name.startswith("cushion"):
                    pool = torch.cat((window_states[:, :BALL], window_states[:, BALL:]))
                else:
                    pool = window_states
                pool = pool[torch.randperm(len(pool), device=args.device)[:500000]]
                module.gate_mean.copy_(pool.mean(0))
                module.gate_std.copy_(pool.std(0).clamp_min(0.01))

        def gate_batch(n):
            """Returns per-module (inputs, targets)."""
            pairs = torch.cat((sample(contact_pairs, n // 3), sample(near_pairs, n // 3), sample(free_pairs, n - 2 * (n // 3))))
            s, _, lab = truth(pairs)
            if model.contact == "scalar_shared":
                return {"any": (s, lab.any(-1).to(s.dtype))}
            if model.contact == "structured4":
                l9 = labels9[pairs[:, 0], pairs[:, 1]].to(s.dtype)
                both = torch.cat((s[:, :BALL], s[:, BALL:]))
                out = {"ab": (s, l9[:, 0])}
                for k, c in enumerate(("xp", "xm", "yp", "ym")):
                    out[f"cushion_{c}"] = (both, torch.cat((l9[:, 1 + k], l9[:, 5 + k])))
                return out
            return {"ab": (s, lab[:, 0].to(s.dtype)),
                    "cushion": (torch.cat((s[:, :BALL], s[:, BALL:])), torch.cat((lab[:, 1], lab[:, 2])).to(s.dtype))}

        gate_params = [p for m in model.modules_.values() for p in m.gate.parameters()]
        opt = torch.optim.Adam(gate_params, lr=cfg.get("gate_lr", 1e-3))
        count = 5 if args.smoke else cfg.get("gate_warmup", 30000)
        for update in range(1, count + 1):
            lr = cfg.get("gate_lr", 1e-3) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * update / count)))
            for group in opt.param_groups:
                group["lr"] = lr
            loss = sum(F.binary_cross_entropy_with_logits(model.modules_[k].logit(x), t) for k, (x, t) in gate_batch(batch * 2).items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if update % 5000 == 0:
                event("gate_adam", update=update, loss=float(loss))
        if cfg.get("gate_lbfgs", 0) and not args.smoke:
            fixed = gate_batch(cfg.get("gate_lbfgs_batch", 120000))
            opt = torch.optim.LBFGS(gate_params, lr=1.0, max_iter=15, history_size=30, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-10, tolerance_change=1e-14)

            def gate_closure():
                opt.zero_grad()
                value = sum(F.binary_cross_entropy_with_logits(model.modules_[k].logit(x), t) for k, (x, t) in fixed.items())
                value.backward()
                return value
            for _ in range(cfg["gate_lbfgs"]):
                opt.step(gate_closure)
        with torch.no_grad():
            report = {}
            for k, (x, t) in gate_batch(60000).items():
                predicted = model.modules_[k].logit(x) >= model.threshold_logit
                truth_on = t > 0.5
                report[k] = dict(fp=int((predicted & ~truth_on).sum()), fn=int((~predicted & truth_on).sum()), positives=int(truth_on.sum()))
        event("gate_fit", **report)
        for p in gate_params:
            p.requires_grad_(False)

        # ---- 3. bounce networks ---------------------------------------
        def response_targets(pairs):
            s, y, lab = truth(pairs)
            with torch.no_grad():
                core = model.core_delta(s, history(pairs))
            if model.contact == "scalar_shared":
                gates = lab.any(-1, keepdim=True).to(s.dtype)
            elif model.contact == "structured4":
                gates = labels9[pairs[:, 0], pairs[:, 1]].to(s.dtype)
            else:
                gates = lab.to(s.dtype)
            return s, y, core, gates

        with torch.no_grad():
            # Per-module input/output normalisation and a data-fitted linear
            # skip, from single-event contact transitions.
            fits = {}
            if model.contact == "scalar_shared":
                s, y, core, _ = response_targets(contact_pairs)
                fits["any"] = (s, y - s - core)
            elif model.contact == "structured4":
                single = labels9.sum(-1) == 1
                p_ab = torch.nonzero(train & single & labels9[..., 0])
                s, y, core, _ = response_targets(p_ab)
                fits["ab"] = (s, y - s - core)
                for k, c in enumerate(("xp", "xm", "yp", "ym")):
                    xs, ts = [], []
                    for j, cols in ((1 + k, slice(0, BALL)), (5 + k, slice(BALL, DIM))):
                        p = torch.nonzero(train & single & labels9[..., j])
                        s, y, core, _ = response_targets(p)
                        xs.append(s[:, cols])
                        ts.append((y - s - core)[:, cols])
                    fits[f"cushion_{c}"] = (torch.cat(xs), torch.cat(ts))
            else:
                single = labels.sum(-1) == 1
                p_ab = torch.nonzero(train & single & labels[..., 0])
                s, y, core, _ = response_targets(p_ab)
                fits["ab"] = (s, y - s - core)
                xs, ts = [], []
                for j, cols in ((1, slice(0, BALL)), (2, slice(BALL, DIM))):
                    p = torch.nonzero(train & single & labels[..., j])
                    s, y, core, _ = response_targets(p)
                    xs.append(s[:, cols])
                    ts.append((y - s - core)[:, cols])
                fits["cushion"] = (torch.cat(xs), torch.cat(ts))
            # Floors on the input spread per channel kind (m, m/s, rad/s): a
            # channel that is nearly constant in contact steps (A's side spin
            # before an A-B hit) must not be magnified into a huge input when a
            # rollout drifts slightly there.
            floor_ball = torch.tensor(cfg.get("bounce_input_floor", [0.01, 0.01, 0.05, 0.05, 1.0, 1.0, 1.0]),
                                      dtype=model.state_mean.dtype, device=args.device)
            for k, (x, target) in fits.items():
                module = model.modules_[k]
                floor = floor_ball.repeat(x.shape[-1] // BALL)
                if len(x) < 2:
                    continue
                module.in_mean.copy_(x.mean(0))
                module.in_std.copy_(torch.maximum(x.std(0), floor))
                module.out_mean.copy_(target.mean(0))
                module.out_std.copy_(target.std(0).clamp_min(1e-4))
                f = (x - module.in_mean) / module.in_std
                # Ridge-regularised linear skip: near-collinear inputs (A's and
                # B's side spin after a hit) otherwise get huge cancelling weights.
                design = torch.cat((f, torch.ones_like(f[:, :1])), -1).cpu()
                y = ((target - module.out_mean) / module.out_std).cpu()
                ridge = cfg.get("skip_ridge", 1e-3) * len(design)
                gram = design.T @ design + ridge * torch.eye(design.shape[1], dtype=design.dtype)
                beta = torch.linalg.solve(gram, design.T @ y).to(args.device)
                module.skip.weight.copy_(beta[:-1].T)
                module.skip.bias.copy_(beta[-1])
                event("bounce_linear_fit", module=k, transitions=len(x))

        bounce_params = [p for m in model.modules_.values() for p in list(m.bounce.parameters()) + list(m.skip.parameters())]
        for p in bounce_params:
            p.requires_grad_(True)

        def bounce_loss(s, y, core, gates):
            prediction = s + core + (gates[..., None] * model.responses(s)).sum(-2)
            return ((prediction - y) / response_scale).square().mean()

        # Responses are also trained on windows just before and after each
        # contact (switch forced on, target = the true change, about zero), so
        # the correction fades to nothing at the edge of the contact region and
        # a switch decision one window early or late stays harmless.
        reach = int(cfg.get("response_near_records", 0))
        module_labels = (labels.any(-1, keepdim=True) if model.contact == "scalar_shared"
                         else labels9 if model.contact == "structured4" else labels)
        forced = module_labels.clone()
        if reach:
            padded = torch.nn.functional.max_pool1d(module_labels.permute(0, 2, 1).to(torch.float32), 2 * reach + 1, stride=1,
                                                    padding=reach).permute(0, 2, 1) > 0.5
            forced = padded | module_labels
        pool_pos = torch.nonzero(train & module_labels.any(-1))
        pool_near = torch.nonzero(train & forced.any(-1) & ~module_labels.any(-1))

        def targets_with_gates(pairs):
            s, y, _ = truth(pairs)
            with torch.no_grad():
                core = model.core_delta(s, history(pairs))
            return s, y, core, forced[pairs[:, 0], pairs[:, 1]].to(s.dtype)

        # The core is frozen: compute its change once for every training window.
        with torch.no_grad():
            def build(pairs):
                parts = [targets_with_gates(chunk) for chunk in pairs.split(65536)]
                return tuple(torch.cat([c[k] for c in parts]) for k in range(4))
            cached_pos = build(pool_pos)
            cached_near = build(pool_near) if len(pool_near) else None
        event("bounce_cache", positives=len(pool_pos), near=len(pool_near))
        bounce_batch = 64 if args.smoke else cfg.get("bounce_batch", batch)
        near_fraction = cfg.get("response_near_fraction", 0.5) if cached_near is not None else 0.0

        def cached_batch(n):
            k = int(n * near_fraction)
            out = []
            for cache, m in ((cached_pos, n - k), (cached_near, k)):
                if m:
                    idx = torch.randint(len(cache[0]), (m,), device=args.device, generator=rng)
                    out.append(tuple(t[idx] for t in cache))
            return tuple(torch.cat([o[j] for o in out]) for j in range(4))

        opt = torch.optim.Adam(bounce_params, lr=cfg.get("bounce_lr", 1e-3))
        count = 5 if args.smoke else cfg.get("bounce_warmup", 20000)
        floor_lr = cfg.get("bounce_lr_floor", 0.001)
        for update in range(1, count + 1):
            lr = cfg.get("bounce_lr", 1e-3) * (floor_lr + (1 - floor_lr) * 0.5 * (1 + math.cos(math.pi * update / count)))
            for group in opt.param_groups:
                group["lr"] = lr
            loss = bounce_loss(*cached_batch(bounce_batch))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bounce_params, 10.0)
            opt.step()
            if update % cfg.get("evaluate_every", 5000) == 0 or update == count:
                check("bounce_adam", update, loss.detach())
        if cfg.get("bounce_lbfgs", 0) and not args.smoke:
            fixed = response_targets(contact_pairs if len(contact_pairs) <= cfg.get("bounce_lbfgs_batch", 400000)
                                     else sample(contact_pairs, cfg.get("bounce_lbfgs_batch", 400000)))
            opt = torch.optim.LBFGS(bounce_params, lr=1.0, max_iter=15, history_size=40, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-10, tolerance_change=1e-14)

            def bounce_closure():
                opt.zero_grad()
                value = bounce_loss(*fixed)
                value.backward()
                return value
            for update in range(1, cfg["bounce_lbfgs"] + 1):
                opt.step(bounce_closure)
                if update % 50 == 0 or update == cfg["bounce_lbfgs"]:
                    check("bounce_lbfgs", update, bounce_closure().detach())
        trainable = bounce_params
    else:
        # Transformer-only control: the core also has to learn the collisions.
        for p in core_params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(core_params, lr=cfg.get("bounce_lr", 1e-4))
        count = 5 if args.smoke else cfg.get("bounce_warmup", 20000)
        for update in range(1, count + 1):
            pairs = torch.cat((sample(contact_pairs, batch // 3), sample(near_pairs, batch // 3), sample(free_pairs, batch - 2 * (batch // 3))))
            s, y, _ = truth(pairs)
            loss = ((model(s, history(pairs)) - y) / response_scale).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core_params, 10.0)
            opt.step()
            if update % cfg.get("evaluate_every", 5000) == 0 or update == count:
                check("transformer_only", update, loss.detach())
        trainable = core_params

    # ---- 4. half-second rollout refinement ------------------------------
    model, _ = load_pool(output / "best.pt", args.device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    # Re-select the same parameter groups on the reloaded model.
    if model.contact != "none":
        trainable = [p for m in model.modules_.values() for p in list(m.bounce.parameters()) + list(m.skip.parameters())]
    else:
        trainable = list(model.backbone.parameters()) + list(model.core_head.parameters())
    for p in trainable:
        p.requires_grad_(True)
    count = 2 if args.smoke else cfg.get("rollout_updates", 0)
    horizon = 4 if args.smoke else round(cfg.get("rollout_horizon_s", 0.5) / model.dt)
    rb = 4 if args.smoke else cfg.get("rollout_batch", 64)
    if count:
        opt = torch.optim.Adam(trainable, lr=cfg.get("rollout_lr", 1e-6))
        for update in range(1, count + 1):
            starts = sample(contact_pairs, rb // 2).clone()
            starts[:, 1] = (starts[:, 1] - torch.randint(1, horizon, (len(starts),), device=args.device, generator=rng) * stride).clamp_min(0)
            pairs = torch.cat((starts, sample(torch.nonzero(train), rb - len(starts))))
            current, hist = states[pairs[:, 0], pairs[:, 1]], history(pairs)
            total, den = current.sum() * 0.0, 0
            for step in range(horizon):
                j = pairs[:, 1] + (step + 1) * stride
                active = j < lengths[pairs[:, 0]]
                y = states[pairs[:, 0], j.clamp_max(states.shape[1] - 1)]
                prediction = model(current, hist)
                total = total + (((prediction - y) / rollout_scale).square().mean(-1) * active).sum()
                den += int(active.sum())
                current = torch.where(active[:, None], prediction, current)
                hist = model.advance_history(hist, current)
            loss = total / max(1, den)
            opt.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(gradient):
                raise RuntimeError("nonfinite rollout gradient")
            opt.step()
            if update % (2 if args.smoke else cfg.get("rollout_evaluate_every", 50)) == 0 or update == count:
                check("half_second", update, loss.detach())
    model, metadata = load_pool(output / "best.pt", args.device)
    report = free_metrics(model, data, val)
    atomic_json(output / "validation.json", report)
    atomic_json(output / "gradient_check.json", launch_gradient_check(model, data, [[2.0, 0.0], [2.6, 0.08], [1.7, -0.11]]))
    done = dict(complete=True, smoke=args.smoke, test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                checkpoint_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
                best_stage=metadata["stage"], best_update=metadata["update"], validation_score=report["selection_score"],
                b_rmse_p95_mm=1000 * report["b_rmse_m"]["p95"], b_at_target_p95_mm=1000 * report["b_at_target_m"]["p95"])
    atomic_json(output / "complete.json", done)
    event("complete", **done)


if __name__ == "__main__":
    main()
