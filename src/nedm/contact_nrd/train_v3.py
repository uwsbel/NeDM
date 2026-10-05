"""Training of the state-only contact networks (version 3) on a frozen version-2 core.

Same recipe as the deployed exact-label runs, with the networks' inputs replaced
by rows of the NRD state (see model_v3):
1. Row normalisation per group (data-fitted mean/std of the coded state history).
2. Collision network: binary cross-entropy on exact per-pair labels, strata per
   pair (1/3 contact windows, 1/3 windows within 50 records of that pair's contact,
   'after' windows excluded, 1/3 random windows); Adam with cosine decay, then an
   L-BFGS polish kept only if a fixed validation BCE drops.
3. Contact network: per-group output normalisation and ridge-fitted linear path from
   single-contact windows; Adam (cosine) on 90 % contact windows with the true
   switches and 10 % far windows with one pair forced on (true change ~ 0). Only
   rows whose switch is on are evaluated.
4. Rollout refinement (Huber on free rollouts + teacher-forced retention).
Checkpoints are scored on validation rollouts; the best is kept. Test never read.
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
from nedm.contact_nrd.model import STATE
from nedm.contact_nrd.model_v2 import load_any
from nedm.contact_nrd.model_v3 import StateContactNRD
from nedm.contact_nrd.train import atomic_json, dilate


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
    contacts = data["contacts"]
    aug = cfg.get("mirror_augment")
    if aug:
        # Training-only copies of every training episode under the system's mirror symmetries:
        # state channels change sign, the pair labels are permuted (e.g. +x cushion <-> -x cushion).
        # Validation (data[...], used by free_metrics) is untouched.
        tr_eps = torch.nonzero(splits == 0).flatten()
        ps, pc, pl, pp = [states], [contacts], [lengths], [splits]
        for signs, perm in aug["transforms"]:
            ps.append(states[tr_eps] * torch.tensor(signs, dtype=states.dtype, device=dev))
            pc.append(contacts[tr_eps][..., perm])
            pl.append(lengths[tr_eps])
            pp.append(torch.zeros_like(splits[tr_eps]))
        states, contacts, lengths, splits = torch.cat(ps), torch.cat(pc), torch.cat(pl), torch.cat(pp)
    n_ep, n_rec, n_mov = states.shape[:3]
    pairs = system["pairs"]
    n_pairs = len(pairs)
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    slot = {k: n for n, k in enumerate(moving)}
    labels = window_labels(contacts, stride)                                           # [N, W, P]
    n_win = labels.shape[1]
    previous = torch.cat((torch.zeros_like(labels[:, :1]), window_labels(contacts, 1)[:, : n_win - 1]), 1)
    after = previous & ~labels
    valid = torch.arange(n_win, device=dev)[None, :] < (lengths - stride)[:, None]
    train = valid & (splits == 0)[:, None]
    vwin = valid & (splits == 1)[:, None]
    val = torch.nonzero(splits == 1).flatten()
    if args.smoke:
        val = val[:24]
    rng = torch.Generator(device=dev).manual_seed(cfg["seed"] + 1000)

    def sample(pool, n):
        return pool[torch.randint(len(pool), (n,), device=dev, generator=rng)]

    def padded_times(t, context):
        raw = t[:, None] + torch.arange(1 - context, 1, device=dev) * stride
        return torch.where(raw < 0, t[:, None] % stride, raw)

    # ---- model: frozen core + state-only contact networks ----------------------
    pretrained = torch.load(cfg["core_checkpoint"], map_location=dev, weights_only=False)
    if pretrained["runtime"]["data_sha256"] != data["index"]["data_sha256"] and cfg.get("core_data_check", True):
        raise ValueError("core trained on different data")
    norm = {key: pretrained["normalization"][key] for key in ("state_mean", "state_std", "delta_scale", "mask")}
    m = cfg["model"]
    newer = m.get("integrate_positions") or m.get("derived_relative") or any(k in m.get(n, {}) for n in ("collision", "contact") for k in ("channels", "blocks"))
    # A new tag when the newer options are on, so an older copy of the code refuses these checkpoints.
    mc = {**m, "architecture": "contact_graph_nrd_v4" if newer else "contact_graph_nrd_v3", "dt_s": data["dt"]}
    model = StateContactNRD(mc, system, norm).to(dev)
    model.load_state_dict({k: v for k, v in pretrained["model_state_dict"].items() if k.startswith(("backbone.", "core_head."))}, strict=False)
    core_params = list(model.backbone.parameters()) + list(model.core_head.parameters())
    for p in core_params:
        p.requires_grad_(False)
    H = max(model.k, model.context)

    def history(windows):
        return states[windows[:, 0, None], padded_times(windows[:, 1], H)]

    # Episode-start padding: report how many contact windows reach it (they should be none).
    first_contact = torch.nonzero(labels.any(-1))
    early = int((first_contact[:, 1] < (H - 1) * stride).sum())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(config=cfg, model_config=mc, system=system["name"], training_episodes=int((splits == 0).sum()),
                   validation_episodes=len(val), test_used_for_selection=False, host=platform.node(),
                   job_id=os.environ["SLURM_JOB_ID"], torch=torch.__version__,
                   parameters={"core": sum(p.numel() for p in core_params),
                               "collision": sum(p.numel() for p in model.collision.parameters()),
                               "contact": sum(p.numel() for p in model.contact.parameters()) + model.contact_skip.numel()},
                   rows=model.rows, groups=model.n_groups, features=model.features, contact_windows_in_padding=early,
                   data_sha256=data["index"]["data_sha256"], core_checkpoint=cfg["core_checkpoint"],
                   source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    atomic_json(output / "run_config.json", runtime)
    start, best = time.perf_counter(), float("inf")

    def channel_scale(values):
        return torch.tensor(values, dtype=torch.float64, device=dev)
    response_scale = channel_scale(cfg["response_scale"])
    rollout_scale = channel_scale(cfg["rollout_scale"])

    def save(path, stage, update, validation=None):
        torch.save(dict(model_config=mc, system=system, normalization=norm, model_state_dict=model.state_dict(),
                        runtime=runtime, stage=stage, update=update, validation=validation), path)

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_s=round(time.perf_counter() - start, 1), **extra)
        with (output / "train_log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    checks = {"count": 0, "last": None}

    def check(stage, update, loss):
        nonlocal best
        was_training = model.training
        model.eval()
        report = free_metrics(model, data, val)
        checks["count"] += 1
        if not math.isfinite(report["selection_score"]):
            report["selection_score"] = 1e9
        checks["last"] = report["selection_score"]
        if report["selection_score"] < best or not (output / "best.pt").exists():
            best = report["selection_score"]
            save(output / "best.pt", stage, update, report)
        save(output / "last.pt", stage, update, report)
        event(stage, update=update, loss=float(loss), rmse_median_mm=1000 * report["rmse_m"]["median"],
              rmse_p95_mm=1000 * report["rmse_m"]["p95"], at_target_median_mm=1000 * report["at_target_m"]["median"],
              at_target_p95_mm=1000 * report["at_target_m"]["p95"], eligible_p95_mm=1000 * report["eligible_at_target_m"]["p95"],
              end_p95_mm=1000 * report["end_m"]["p95"], events_ok=report["event_ok_fraction"], score=report["selection_score"])
        if was_training:
            model.train()

    # ---- 1. per-group input maps (training data only) ---------------------------
    width = model.row_width // STATE
    block = torch.tensor([0.01] * 3 + [0.05] * 3 + [1.0] * 3, dtype=torch.float64, device=dev)
    dblock = torch.tensor([1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3, dtype=torch.float64, device=dev)
    floor = torch.cat([block.repeat(width)] + [dblock.repeat(width)] * (model.k - 1)
                      + ([torch.tensor([1e-3] * 3 + [1e-2] * 3, dtype=torch.float64, device=dev)] if model.n_derived else []))
    whiten_floor = cfg.get("whiten_floor", 1e-4)

    def affine_map(xs, cols):
        """Rows [n, F] (columns `cols` of the coded row) -> (mean, projection): standardise with floors,
        then (optionally) whiten with an eigenvalue floor (ZCA)."""
        mean = xs.mean(0)
        std = torch.maximum(xs.std(0), floor[cols])
        z = (xs - mean) / std
        if not cfg.get("whiten", True):
            return mean, torch.diag(1.0 / std)
        cov = (z.T @ z) / max(1, len(z) - 1)
        lam, vec = torch.linalg.eigh(cov)
        w = vec @ torch.diag(lam.clamp_min(whiten_floor).rsqrt()) @ vec.T
        return mean, torch.diag(1.0 / std) @ w

    def rows_of(windows, row_mask):
        """Coded rows of the given (window, row) pairs: windows [n, 2], row_mask [n, R] -> [m, F], group [m]."""
        x = torch.cat([model.row_inputs(history(c)) for c in windows.split(32768)])
        idx = torch.nonzero(row_mask)
        return x[idx[:, 0], idx[:, 1]], model.row_group[idx[:, 1]]

    near_rec = cfg.get("near_negative_records", 50)
    near_pair = dilate(labels, near_rec) & train[..., None]                                 # [N, W, P]
    fit_report = {}
    with torch.no_grad():
        rand_w = sample(torch.nonzero(train), 50000)
        rx, rg = rows_of(rand_w, torch.ones(len(rand_w), model.n_rows, dtype=torch.bool, device=dev))
        near_w = sample(torch.nonzero(near_pair.any(-1)), 200000)
        nx, ng = rows_of(near_w, near_pair[near_w[:, 0], near_w[:, 1]][:, model.row_pair])
        cg = model.cols_gate
        for g in range(model.n_groups):
            xs = nx[ng == g][:, cg]
            if len(xs) < 10 * model.f_gate:
                xs = torch.cat((xs, rx[rg == g][:, cg]))
            model.gate_mean3[g], model.gate_proj3[g] = affine_map(xs, cg)
            fit_report[f"group{g}"] = dict(near_rows=int((ng == g).sum()))
        norm["gate_mean3"], norm["gate_proj3"] = model.gate_mean3.tolist(), model.gate_proj3.tolist()
    event("gate_input_map", whiten=cfg.get("whiten", True), whiten_floor=whiten_floor, contact_windows_in_padding=early, **fit_report)

    def scaled(windows, which):
        return model.scaled_rows(history(windows), which)

    # ---- 2. collision network ----------------------------------------------------
    near = dilate(labels, cfg.get("near_negative_records", 50)) & ~labels & ~after
    strata = []
    for p in range(n_pairs):
        strata.append((p, torch.nonzero(labels[..., p] & train), torch.nonzero(near[..., p] & train),
                       torch.nonzero(labels[..., p] & vwin), torch.nonzero(near[..., p] & vwin)))
    any_train, any_val = torch.nonzero(train), torch.nonzero(vwin)

    def collision_batch(n, split="train"):
        out = []
        for p, pos, neg, vpos, vneg in strata:
            pos, neg, anyw = (pos, neg, any_train) if split == "train" else (vpos, vneg, any_val)
            m = max(3, n // n_pairs)
            parts = [sample(pos, m // 3)] if len(pos) else []
            parts += [sample(neg, m // 3)] if len(neg) else []
            parts.append(sample(anyw, m - sum(len(q) for q in parts)))
            wnd = torch.cat(parts)
            out.append(torch.cat((wnd, torch.full_like(wnd[:, :1], p)), -1))
        return torch.cat(out)

    def collision_loss(picks):
        total = 0.0
        for c in picks.split(65536):
            z = scaled(c[:, :2], "gate")
            logit = model.pair_logits(z)[torch.arange(len(c), device=dev), c[:, 2]]
            target = labels[c[:, 0], c[:, 1], c[:, 2]].to(torch.float64)
            total = total + F.binary_cross_entropy_with_logits(logit, target, reduction="sum")
        return total / len(picks)

    gate_params = list(model.collision.parameters())
    gate_lr = cfg.get("gate_lr", 1e-3)
    opt = torch.optim.Adam(gate_params, lr=gate_lr)
    count = 5 if args.smoke else cfg.get("gate_updates", 30000)
    warm = cfg.get("gate_warmup", 0)
    for update in range(1, count + 1):
        for group in opt.param_groups:
            group["lr"] = gate_lr * min(1.0, update / warm if warm else 1.0) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * update / count)))
        loss = collision_loss(collision_batch(cfg.get("gate_batch", 2048)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 10.0)
        opt.step()
        if update % 10000 == 0:
            event("collision_adam", update=update, loss=float(loss))
    with torch.no_grad():
        val_picks = collision_batch(60000, "val")
        val_bce = float(collision_loss(val_picks))
    event("collision_val_bce", bce=val_bce)
    if cfg.get("gate_lbfgs", 0) and not args.smoke:
        keep = {k: v.clone() for k, v in model.collision.state_dict().items()}
        fixed = collision_batch(cfg.get("gate_lbfgs_batch", 120000))
        opt = torch.optim.LBFGS(gate_params, lr=1.0, max_iter=15, history_size=30, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-14)

        def gate_closure():
            opt.zero_grad()
            value = collision_loss(fixed)
            value.backward()
            return value
        t0 = time.perf_counter()
        for _ in range(cfg["gate_lbfgs"]):
            opt.step(gate_closure)
            if time.perf_counter() - t0 > cfg.get("gate_lbfgs_cap_s", 1800):
                break
        with torch.no_grad():
            after_bce = float(collision_loss(val_picks))
        kept = after_bce < val_bce
        if not kept:
            model.collision.load_state_dict(keep)
        event("collision_lbfgs", val_bce_before=val_bce, val_bce_after=after_bce, kept=kept)
    with torch.no_grad():
        report = {}
        for p in range(n_pairs):
            sel = val_picks[val_picks[:, 2] == p]
            z = scaled(sel[:, :2], "gate")
            on = model.pair_logits(z)[torch.arange(len(sel), device=dev), p] >= 0
            truth = labels[sel[:, 0], sel[:, 1], p]
            report[f"pair{p}"] = dict(fp=int((on & ~truth).sum()), fn=int((~on & truth).sum()), positives=int(truth.sum()))
        event("collision_fit", **report)
    for p in gate_params:
        p.requires_grad_(False)

    # ---- 3. contact network ------------------------------------------------------
    pos_windows = torch.nonzero(train & labels.any(-1))
    far_mask = train & ~dilate(labels, cfg.get("far_records", 50)).any(-1)
    far_windows = sample(torch.nonzero(far_mask), cfg.get("far_cache", 200000))

    def gates_for(windows, far=False):
        if far:
            g = torch.zeros(len(windows), n_pairs, dtype=torch.float64, device=dev)
            g[torch.arange(len(windows), device=dev), torch.randint(n_pairs, (len(windows),), device=dev, generator=rng)] = 1.0
            return g
        return labels[windows[:, 0], windows[:, 1]].to(torch.float64)

    def core_of(windows):
        with torch.no_grad():
            return torch.cat([model.core_delta(states[c[:, 0], c[:, 1]], history(c)) for c in windows.split(32768)])

    caches = {"pos": (pos_windows, gates_for(pos_windows), core_of(pos_windows)),
              "far": (far_windows, gates_for(far_windows, True), core_of(far_windows))}
    event("contact_cache", positive_windows=len(pos_windows), far_windows=len(far_windows))
    with torch.no_grad():
        fw = far_windows[:20000]
        s0, s1, c0 = states[fw[:, 0], fw[:, 1]], states[fw[:, 0], fw[:, 1] + stride], caches["far"][2][:20000]
        trap = s1[..., 0:3] - s0[..., 0:3] - model.integrated_dp(s0[..., 3:6], s1[..., 3:6])
        corep = s1[..., 0:3] - s0[..., 0:3] - c0[..., 0:3]
        event("free_step_position_error", trapezoid_true_velocity_p95_m=float(torch.quantile(trap.norm(dim=-1).flatten().float(), 0.95)),
              core_p95_m=float(torch.quantile(corep.norm(dim=-1).flatten().float(), 0.95)), integrate=model.integrate)

    with torch.no_grad():
        wins, g, core = caches["pos"]
        single = g.sum(-1) == 1
        wins, g, core = wins[single], g[single], core[single]
        resid = model.contact_target(states[wins[:, 0], wins[:, 1]], states[wins[:, 0], wins[:, 1] + stride], core)   # [n, D, 9]
        rows_on = g[:, model.row_pair] > 0.5                                                  # [n, R]
        idx = torch.nonzero(rows_on)
        xr, grp = rows_of(wins, rows_on)                                                      # raw coded rows
        # Contact input map per group, from that group's contact rows (groups never in contact: near/random rows).
        cq = model.cols_cont
        for gi in range(model.n_groups):
            xs = xr[grp == gi][:, cq]
            if len(xs) < 10 * model.f_cont:
                xs = torch.cat((xs, nx[ng == gi][:, cq], rx[rg == gi][:, cq]))
            model.cont_mean3[gi], model.cont_proj3[gi] = affine_map(xs, cq)
        norm["cont_mean3"], norm["cont_proj3"] = model.cont_mean3.tolist(), model.cont_proj3.tolist()
        zr = model.squash(torch.einsum("mf,mfg->mg", xr[:, cq] - model.cont_mean3[grp], model.cont_proj3[grp]))
        if model.routing == "pair":   # each routed row's target is its own affected body's change (9)
            tgt = resid[idx[:, 0], model.row_affects[idx[:, 1]].argmax(-1)]
        else:
            tgt = (resid[idx[:, 0]] * model.row_affects[idx[:, 1]][..., None]).reshape(len(idx), -1)
        fit_report, missing = {}, []
        for gi in range(model.n_groups):
            sel = grp == gi
            if sel.sum() < 2:
                missing.append(gi)
                continue
            xt, yt = zr[sel], tgt[sel]
            model.out_mean3[gi] = yt.mean(0)
            model.out_std3[gi] = yt.std(0).clamp_min(1e-4)
            design = torch.cat((xt, torch.ones_like(xt[:, :1])), -1).cpu()
            yy = ((yt - model.out_mean3[gi]) / model.out_std3[gi]).cpu()
            gram = design.T @ design + cfg.get("skip_ridge", 1e-3) * len(design) * torch.eye(design.shape[1], dtype=design.dtype)
            beta = torch.linalg.solve(gram, design.T @ yy).to(dev)
            model.contact_skip.data[gi] = beta
            fit_report[f"group{gi}"] = dict(rows=int(sel.sum()), max_skip_weight=float(beta[:-1].abs().max()))
        present = [gi for gi in range(model.n_groups) if gi not in missing]
        for gi in missing:   # a group never in contact: typical scale, zero mean and path; it only learns "no change"
            model.out_std3[gi] = model.out_std3[present].mean(0)
        norm["out_mean3"], norm["out_std3"] = model.out_mean3.tolist(), model.out_std3.tolist()
    event("contact_linear_fit", groups_without_contacts=missing, **fit_report)

    contact_params = list(model.contact.parameters()) + [model.contact_skip]
    mix = cfg.get("contact_mix", {"pos": 0.9, "far": 0.1})

    def mixture(n):
        out = []
        for name, share in mix.items():
            wins, g, core = caches[name]
            m = int(round(n * share))
            if m == 0:
                continue
            idx = torch.randint(len(wins), (m,), device=dev, generator=rng)
            out.append((wins[idx], g[idx], core[idx]))
        return tuple(torch.cat([o[j] for o in out]) for j in range(3))

    rigid = cfg.get("pair_rigid_augment")   # {"pair": p, "prob": q, "half_extent": [hx, hy]}

    def rigid_motion(x, cos, sin, centre, target):
        """Rotate states [n, ..., D, 9] about the vertical through `centre` [n, 2] and move that point to `target`."""
        out = x.clone()
        shape = (-1,) + (1,) * (x.dim() - 2)
        c, s_, cx, cy, tx, ty = (v.reshape(shape) for v in (cos, sin, centre[:, 0], centre[:, 1], target[:, 0], target[:, 1]))
        px, py = x[..., 0] - cx, x[..., 1] - cy
        out[..., 0], out[..., 1] = c * px - s_ * py + tx, s_ * px + c * py + ty
        for k in (3, 6):   # velocity and spin rotate as vectors about the vertical
            out[..., k], out[..., k + 1] = c * x[..., k] - s_ * x[..., k + 1], s_ * x[..., k] + c * x[..., k + 1]
        return out

    def contact_loss(wins, g, core):
        s, y = states[wins[:, 0], wins[:, 1]], states[wins[:, 0], wins[:, 1] + stride]
        hist = history(wins)
        if rigid is not None:
            # Training-only copies of single A-B contact steps under a rigid motion of the pair on the table
            # (the collision itself does not depend on where or in which direction it happens).
            sel = (g[:, rigid["pair"]] > 0.5) & (g.sum(-1) == 1)
            sel &= torch.rand(len(g), device=dev, generator=rng) < rigid.get("prob", 0.5)
            idx = torch.nonzero(sel).flatten()
            if len(idx):
                n = len(idx)
                theta = torch.rand(n, device=dev, generator=rng, dtype=s.dtype) * 2 * math.pi
                centre = s[idx, :, 0:2].mean(1)
                half = torch.tensor(rigid["half_extent"], device=dev, dtype=s.dtype)
                target = (torch.rand(n, 2, device=dev, generator=rng, dtype=s.dtype) * 2 - 1) * half
                cos, sin = theta.cos(), theta.sin()
                s, y, hist, core = s.clone(), y.clone(), hist.clone(), core.clone()
                s[idx] = rigid_motion(s[idx], cos, sin, centre, target)
                y[idx] = rigid_motion(y[idx], cos, sin, centre, target)
                hist[idx] = rigid_motion(hist[idx], cos, sin, centre, target)
                with torch.no_grad():
                    core[idx] = model.core_delta(s[idx], hist[idx])
        z = model.scaled_rows(hist, "contact")
        pred = model.next_state(s, core, model.contact_sum(z, g[:, model.row_pair]))
        inv = ((g[:, model.row_pair, None] * model.row_affects).amax(1) > 0).to(torch.float64)      # [n, D]
        err = ((pred - y) * model.mask / response_scale).square().mean(-1)
        return (err * inv).sum() / inv.sum().clamp_min(1)

    for p in contact_params:
        p.requires_grad_(True)
    lr0 = cfg.get("contact_lr", 1e-3)
    wd = cfg.get("contact_weight_decay", 0.0)
    opt = torch.optim.AdamW(contact_params, lr=lr0, weight_decay=wd) if wd else torch.optim.Adam(contact_params, lr=lr0)
    count = 5 if args.smoke else cfg.get("contact_updates", 160000)
    floor_lr = cfg.get("contact_lr_floor", 0.01)
    warm = cfg.get("contact_warmup", 0)
    for update in range(1, count + 1):
        for group in opt.param_groups:
            group["lr"] = lr0 * min(1.0, update / warm if warm else 1.0) * (floor_lr + (1 - floor_lr) * 0.5 * (1 + math.cos(math.pi * update / count)))
        loss = contact_loss(*mixture(64 if args.smoke else cfg.get("contact_batch", 4096)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(contact_params, 10.0)
        opt.step()
        if update % (5 if args.smoke else cfg.get("evaluate_every", 10000)) == 0 or update == count:
            check("contact_adam", update, loss.detach())

    # ---- 4. rollout refinement ---------------------------------------------------
    phases = cfg.get("refine_phases", [])
    if cfg.get("futility_score") and checks["last"] is not None and best > cfg["futility_score"]:
        event("futility_skip_refinement", best=best, bar=cfg["futility_score"])
        phases = []
    model, _ = load_any(output / "best.pt", dev)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    contact_params = list(model.contact.parameters()) + [model.contact_skip]
    for p in contact_params:
        p.requires_grad_(True)
    contact_windows = torch.nonzero(train & labels.any(-1))
    any_window = torch.nonzero(train)
    for phase, spec in enumerate(phases):
        count = 2 if args.smoke else spec["updates"]
        horizon = 4 if args.smoke else round(spec["horizon_s"] / model.dt)
        rb = 4 if args.smoke else spec.get("batch", 256)
        opt = torch.optim.AdamW(contact_params, lr=spec["lr"], weight_decay=wd) if wd else torch.optim.Adam(contact_params, lr=spec["lr"])
        for update in range(1, count + 1):
            starts = sample(contact_windows, rb // 2).clone()
            starts[:, 1] = (starts[:, 1] - torch.randint(1, horizon, (len(starts),), device=dev, generator=rng) * stride).clamp_min(0)
            w = torch.cat((starts, sample(any_window, rb - len(starts))))
            current = states[w[:, 0], w[:, 1]]
            hist = history(w)
            total, den = current.sum() * 0.0, 0
            for step in range(horizon):
                j = w[:, 1] + (step + 1) * stride
                active = j < lengths[w[:, 0]]
                y = states[w[:, 0], j.clamp_max(n_rec - 1)]
                pred = model(current, hist)
                err = F.huber_loss((pred - y) * model.mask / rollout_scale, torch.zeros_like(pred), reduction="none", delta=1.0)
                total = total + (err.mean((-1, -2)) * active).sum()
                den += int(active.sum())
                current = torch.where(active[:, None, None], pred, current)
                hist = model.advance_history(hist, current)
            loss = total / max(1, den)
            if spec.get("retention", 1.0) > 0:
                loss = loss + spec.get("retention", 1.0) * contact_loss(*mixture(cfg.get("contact_batch", 4096) // 4))
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(contact_params, 1.0)
            if not torch.isfinite(gnorm):
                raise RuntimeError("nonfinite rollout gradient")
            opt.step()
            if update % (2 if args.smoke else spec.get("check_every", 100)) == 0 or update == count:
                check(f"rollout{phase}", update, loss.detach())
                if time.perf_counter() - start > cfg.get("refine_deadline_s", 13500):
                    event("refine_deadline", phase=phase, update=update)
                    break
        else:
            continue
        break
    model, metadata = load_any(output / "best.pt", dev)
    report = free_metrics(model, data, val)
    atomic_json(output / "validation.json", report)
    done = dict(complete=True, smoke=args.smoke, test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                checkpoint_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(), best_stage=metadata["stage"],
                best_update=metadata["update"], validation_score=report["selection_score"], checkpoints_compared=checks["count"],
                at_target_p95_mm=1000 * report["at_target_m"]["p95"], eligible_p95_mm=1000 * report["eligible_at_target_m"]["p95"],
                rmse_p95_mm=1000 * report["rmse_m"]["p95"], rmse_median_mm=1000 * report["rmse_m"]["median"],
                end_p95_mm=1000 * report["end_m"]["p95"], events_ok=report["event_ok_fraction"])
    atomic_json(output / "complete.json", done)
    event("complete", **done)


if __name__ == "__main__":
    main()
