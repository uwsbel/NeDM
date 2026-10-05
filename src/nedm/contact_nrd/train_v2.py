"""Staged, validation-only training of the unified contact NRD, version 2.

1. Core (each moving body, steps without its own contact): sampling over
   log-spaced bins of the velocity change with weight ~ sqrt(count) (bodies
   exactly at rest left out when anchoring); data-fitted readout; Adam phases
   with an L-BFGS polish in between.
2. Collision network: binary cross-entropy in strata by pair type (positives
   on labels widened by one model step each side, same-pair near negatives,
   random pairs); sphere-sphere trains the averaged logit; Adam + L-BFGS.
3. Contact network: per-type normalisation and ridge-fitted linear skips from
   single-contact windows; Adam on a mixture of contact windows (true
   switches), near-contact band windows (that pair forced on; optional
   approach-side gap) and far windows (one pair forced on, true change ~ 0);
   then an L-BFGS polish on a fixed sample of the same mixture.
4. Rollout refinement phases with a Huber loss and a teacher-forced retention
   term. Checkpoints are scored on validation rollouts throughout and the best
   is kept. The test split is never read.
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
from nedm.contact_nrd.model_v2 import FEATURES, TYPES, ContactGraphNRDv2, load_any
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
    n_ep, n_rec, n_mov = states.shape[:3]
    pairs = system["pairs"]
    n_pairs = len(pairs)
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    slot = {k: n for n, k in enumerate(moving)}
    pair_type = torch.tensor([0 if system["bodies"][j]["kind"] == "sphere" else 1 for _, j in pairs], device=dev)
    contacts = data["contacts"]
    labels = window_labels(contacts, stride)                                           # [N, W, P]
    n_win = labels.shape[1]
    previous = torch.cat((torch.zeros_like(labels[:, :1]), window_labels(contacts, 1)[:, : n_win - 1]), 1)
    after = previous & ~labels
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

    def history(windows, context):
        return states[windows[:, 0, None], padded_times(windows[:, 1], context)]

    # ---- state normalisation ------------------------------------------------
    sw = torch.nonzero(train)
    sw = sw[torch.randperm(len(sw), device=dev, generator=rng)[:200000]]
    s0, s1 = states[sw[:, 0], sw[:, 1]], states[sw[:, 0], sw[:, 1] + stride]
    free = ~involved[sw[:, 0], sw[:, 1]]
    flat0, flatd = s0.reshape(-1, STATE), (s1 - s0).reshape(-1, STATE)
    keep = free.reshape(-1)
    floors = torch.tensor([0.01] * 3 + [0.01] * 3 + [0.1] * 3, dtype=torch.float64, device=dev)
    state_std = flat0.std(0)
    span = lambda x: (x.quantile(0.999, dim=0) - x.quantile(0.001, dim=0)) if len(x) < 16000000 else x.std(0) * 6
    rng_states = span(flat0[torch.randperm(len(flat0), device=dev)[:100000]])
    # Data rule for constant channels: p99.9-p0.1 range below 1 % of the largest live range of the same unit.
    mask = torch.ones(STATE, dtype=torch.float64, device=dev)
    for cols in ((0, 1, 2), (3, 4, 5), (6, 7, 8)):
        biggest = rng_states[list(cols)].max()
        for c in cols:
            if rng_states[c] < 0.01 * biggest:
                mask[c] = 0.0
    norm = dict(state_mean=flat0.mean(0).tolist(), state_std=torch.maximum(state_std, floors).tolist(),
                delta_scale=flatd[keep].std(0).clamp_min(1e-6).tolist(), mask=mask.tolist())
    mc = {**cfg["model"], "architecture": "contact_graph_nrd_v2", "dt_s": data["dt"]}
    pretrained = None
    if cfg.get("init_checkpoint") or cfg.get("core_checkpoint"):
        pretrained = torch.load(cfg.get("init_checkpoint") or cfg["core_checkpoint"], map_location=dev, weights_only=False)
        if pretrained["runtime"]["data_sha256"] != data["index"]["data_sha256"]:
            raise ValueError("checkpoint trained on different data")
        for key in ("state_mean", "state_std", "delta_scale", "mask"):
            norm[key] = pretrained["normalization"][key]
        if cfg.get("init_checkpoint"):
            norm = dict(pretrained["normalization"])
    model = ContactGraphNRDv2(mc, system, norm).to(dev)
    if cfg.get("init_checkpoint"):
        model.load_state_dict(pretrained["model_state_dict"])
    elif cfg.get("core_checkpoint"):
        model.load_state_dict({k: v for k, v in pretrained["model_state_dict"].items() if k.startswith(("backbone.", "core_head."))}, strict=False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(config=cfg, model_config=mc, system=system["name"], training_episodes=int((splits == 0).sum()),
                   validation_episodes=len(val), test_used_for_selection=False, host=platform.node(),
                   job_id=os.environ["SLURM_JOB_ID"], torch=torch.__version__, mask=mask.tolist(),
                   parameters=sum(p.numel() for p in model.parameters()), data_sha256=data["index"]["data_sha256"],
                   source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    atomic_json(output / "run_config.json", runtime)
    start, best = time.perf_counter(), float("inf")

    def channel_scale(values):
        return torch.tensor(values, dtype=torch.float64, device=dev)
    free_scale = channel_scale(cfg["free_scale"])
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

    checks = {"count": 0}

    def check(stage, update, loss):
        nonlocal best
        was_training = model.training
        model.eval()
        report = free_metrics(model, data, val)
        checks["count"] += 1
        if not math.isfinite(report["selection_score"]):
            report["selection_score"] = 1e9
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

    core_params = list(model.backbone.parameters()) + list(model.core_head.parameters())

    # ---- 1. core --------------------------------------------------------------
    if pretrained is None:
        phase_keep = (torch.arange(n_win, device=dev) % cfg.get("free_phase_step", 5) == 0)[None, :]
        triples = []
        for b in range(n_mov):
            w = torch.nonzero(train & phase_keep & ~involved[..., b])
            triples.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
        triples = torch.cat(triples)
        with torch.no_grad():
            dvel = torch.cat([(states[c[:, 0], c[:, 1] + stride, c[:, 2], 3:6] - states[c[:, 0], c[:, 1], c[:, 2], 3:6]).norm(dim=-1)
                              for c in triples.split(1 << 20)])
            speed = torch.cat([states[c[:, 0], c[:, 1], c[:, 2], 3:9].abs().amax(-1) for c in triples.split(1 << 20)])
            moving_step = speed > 1e-6 if model.anchor_rest else torch.ones_like(speed, dtype=torch.bool)
            triples, dvel = triples[moving_step], dvel[moving_step]
            logd = torch.log10(dvel.clamp_min(1e-12))
            sub = logd[torch.randperm(len(logd), device=dev)[:1000000]].float()
            lo, hi = torch.quantile(sub, 0.001).double(), torch.quantile(sub, 0.999).double()
            bins = cfg.get("core_bins", 8)
            if hi - lo < 1e-6:
                which = torch.zeros_like(logd, dtype=torch.long)
            else:
                which = ((logd - lo) / (hi - lo) * bins).floor().clamp(0, bins - 1).long()
            groups = [triples[which == k] for k in range(bins) if (which == k).any()]
            weights = torch.tensor([math.sqrt(len(g)) for g in groups], dtype=torch.float64)
            weights = (weights / weights.sum()).tolist()
        runtime["core_bins"] = [{"count": len(g), "weight": w} for g, w in zip(groups, weights)]
        atomic_json(output / "run_config.json", runtime)
        up = model.up.double()

        def rotate_about_up(v, angle):
            c, s = angle.cos(), angle.sin()
            while c.dim() < v.dim():
                c, s = c[..., None], s[..., None]
            k = up.expand_as(v)
            return v * c + torch.cross(k, v, dim=-1) * s + k * (k * v).sum(-1, keepdim=True) * (1 - c)

        def core_batch(n, augment=True):
            tr = torch.cat([sample(g, max(1, int(round(n * w)))) for g, w in zip(groups, weights)])
            rows = states[tr[:, 0, None], padded_times(tr[:, 1], model.context), tr[:, 2, None]]
            target = states[tr[:, 0], tr[:, 1] + stride, tr[:, 2]] - rows[:, -1]
            if augment and cfg.get("rotation_augment", False):
                angle = torch.rand(len(tr), device=dev, generator=rng, dtype=torch.float64) * 2 * math.pi
                rows = torch.cat([rotate_about_up(rows[..., k:k + 3], angle) for k in (0, 3, 6)], -1)
                target = torch.cat([rotate_about_up(target[..., k:k + 3], angle) for k in (0, 3, 6)], -1)
            return rows, target * model.mask

        hist, target = core_batch(128 if args.smoke else 32768, augment=False)
        with torch.no_grad():
            emb = model.backbone((hist - model.state_mean) / model.state_std)[..., -1, :]
            if model.anchor_rest:
                rest = torch.cat((hist[..., :3], torch.zeros_like(hist[..., 3:])), -1)
                emb = emb - model.backbone((rest - model.state_mean) / model.state_std)[..., -1, :]
                design = emb
            else:
                design = torch.cat((emb, torch.ones_like(emb[:, :1])), -1)
            beta = torch.linalg.lstsq(design.cpu(), (target / model.delta_scale).cpu(), rcond=cfg.get("readout_rcond", 1e-2),
                                      driver="gelsd").solution.to(dev)
            model.core_head.weight.copy_(beta[:emb.shape[1]].T)
            model.core_head.bias.copy_(torch.zeros_like(model.core_head.bias) if model.anchor_rest else beta[emb.shape[1]])
        event("core_readout_fit", bins=runtime["core_bins"])

        def core_loss(hist, target):
            return ((model.body_delta(hist) - target) / free_scale).square().mean()

        # Held-out free-flight windows: every phase is scored on them and the best phase is kept
        # (amendment 1: an Adam phase after L-BFGS can undo it).
        with torch.no_grad():
            pv = []
            for b in range(n_mov):
                w = torch.nonzero(valid & (splits == 1)[:, None] & ~involved[..., b])
                w = w[torch.randperm(len(w), device=dev, generator=rng)[:100000]]
                pv.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
            pv = torch.cat(pv)
            pv_hist = states[pv[:, 0, None], padded_times(pv[:, 1], model.context), pv[:, 2, None]]
            pv_target = (states[pv[:, 0], pv[:, 1] + stride, pv[:, 2]] - pv_hist[:, -1]) * model.mask
            pv_moving = pv_hist[:, -1, 3:9].abs().amax(-1) > 1e-6

        def core_val():
            with torch.no_grad():
                err = torch.cat([model.body_delta(h) for h in pv_hist.split(16384)]) - pv_target
                return err[pv_moving], float((err[pv_moving] / free_scale).square().mean())

        core_keys = [k for k in model.state_dict() if k.startswith(("backbone.", "core_head."))]
        best_core = {"score": core_val()[1], "phase": "readout", "state": {k: model.state_dict()[k].clone() for k in core_keys}}
        event("core_phase_check", phase="readout", val_loss=best_core["score"])
        loss = torch.zeros(())
        for phase, spec in enumerate(cfg["core_phases"]):
            if spec.get("lbfgs"):
                if args.smoke:
                    continue
                hist, target = core_batch(spec.get("batch", 65536))
                opt = torch.optim.LBFGS(core_params, lr=1.0, max_iter=15, history_size=30, line_search_fn="strong_wolfe",
                                        tolerance_grad=1e-12, tolerance_change=1e-15)

                def closure():
                    opt.zero_grad()
                    value = core_loss(hist, target)
                    value.backward()
                    return value
                for _ in range(spec["lbfgs"]):
                    opt.step(closure)
                event(f"core_phase{phase}_lbfgs", loss=float(closure()))
                score = core_val()[1]
                event("core_phase_check", phase=phase, val_loss=score)
                if score < best_core["score"]:
                    best_core = {"score": score, "phase": phase, "state": {k: model.state_dict()[k].clone() for k in core_keys}}
                continue
            opt = torch.optim.Adam(core_params, lr=spec["lr"])
            updates = 5 if args.smoke else spec["updates"]
            for update in range(1, updates + 1):
                for group in opt.param_groups:
                    group["lr"] = spec["lr"] * (0.001 + 0.999 * 0.5 * (1 + math.cos(math.pi * update / updates)))
                loss = core_loss(*core_batch(64 if args.smoke else spec["batch"]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core_params, 10.0)
                opt.step()
                if update % 10000 == 0:
                    event(f"core_phase{phase}", update=update, loss=float(loss))
            score = core_val()[1]
            event("core_phase_check", phase=phase, val_loss=score)
            if score < best_core["score"]:
                best_core = {"score": score, "phase": phase, "state": {k: model.state_dict()[k].clone() for k in core_keys}}
        model.load_state_dict(best_core["state"], strict=False)
        runtime["core_kept_phase"] = best_core["phase"]
        runtime["core_kept_val_loss"] = best_core["score"]
        atomic_json(output / "run_config.json", runtime)
        event("core_kept", phase=best_core["phase"], val_loss=best_core["score"])
        save(output / "free_core.pt", "free_core", 0)
        with torch.no_grad():
            err = core_val()[0]
            core_report = {name: {q: float(torch.quantile(err[:, cols].norm(dim=-1), qq)) for q, qq in (("median", 0.5), ("p95", 0.95))}
                           for name, cols in (("pos_m", [0, 1, 2]), ("vel_mps", [3, 4, 5]), ("spin_radps", [6, 7, 8]))}
        atomic_json(output / "core_validation.json", core_report)
        event("core_validation", **{k: v["p95"] for k, v in core_report.items()})
        if cfg.get("core_only"):
            atomic_json(output / "complete.json", dict(complete=True, core_only=True, smoke=args.smoke,
                        total_elapsed_s=time.perf_counter() - start, core_validation=core_report))
            event("complete_core_only")
            return
    for p in core_params:
        p.requires_grad_(False)
    contact_params = list(model.contact.parameters()) + list(model.contact_skip.parameters())

    reach = cfg.get("response_near_records", 30)
    if not cfg.get("init_checkpoint"):
        # ---- pair-feature normalisation (training data only) -------------------
        band = dilate(labels, reach) & train[..., None]
        with torch.no_grad():
            model.compress.fill_(1e6)
            bw = torch.nonzero(band.any(-1))
            bw = bw[torch.randperm(len(bw), device=dev, generator=rng)[:200000]]
            f_band, _, b_band, _, mm = model.all_pair_features(states[bw[:, 0], bw[:, 1]])
            in_band = band[bw[:, 0], bw[:, 1]]                                               # [M, P]
            for t in range(TYPES):
                cols = pair_type == t
                if not cols.any():
                    continue
                d = f_band[:, cols, :3].norm(dim=-1)[in_band[:, cols]]
                model.compress[t] = float(torch.quantile(d.float(), 0.99)) if len(d) else 1.0
            model.compress.clamp_(min=0.01)
            feat_floor = torch.tensor([0.01] * 3 + [0.05] * 3 + [1.0] * 3 + [0.05] * 3 + [1.0] * 3 + [1e-3] * 3 + [1e-3] * 2,
                                      dtype=torch.float64, device=dev)
            forward, _, backward, _, mm = model.all_pair_features(s0)
            for t in range(TYPES):
                cols = torch.nonzero(pair_type == t).flatten()
                if not len(cols):
                    continue
                x = forward[:, cols].reshape(-1, FEATURES)
                model.gate_mean2[t] = x.mean(0)
                model.gate_std2[t] = torch.maximum(x.std(0), feat_floor)
        event("pair_normalisation", compress=model.compress.tolist())

        # ---- 2. collision network --------------------------------------------
        widen = cfg.get("collision_widen_records", stride)
        coll_labels = dilate(labels, widen) if widen else labels
        near = dilate(labels, cfg.get("near_negative_records", 50)) & ~coll_labels & ~after & train[..., None]
        strata = []
        for t in range(TYPES):
            tp = (pair_type == t)
            if not tp.any():
                continue
            pos = torch.nonzero(coll_labels & train[..., None] & tp)
            neg = torch.nonzero(near & tp)
            strata.append((t, pos, neg, torch.nonzero(tp).flatten()))
        any_window = torch.nonzero(train)

        def collision_batch(n):
            out = []
            for t, pos, neg, type_pairs in strata:
                m = n // len(strata)
                rand = sample(any_window, m - 2 * (m // 3))
                rand = torch.cat((rand, type_pairs[torch.randint(len(type_pairs), (len(rand), 1), device=dev, generator=rng)]), -1)
                picks = torch.cat((sample(pos, m // 3), sample(neg, m // 3), rand))
                out.append(picks)
            return out

        def collision_loss(batches):
            total = 0.0
            for picks in batches:
                s = states[picks[:, 0], picks[:, 1]]
                target = coll_labels[picks[:, 0], picks[:, 1], picks[:, 2]].to(torch.float64)
                forward, _, backward, _, mm = model.all_pair_features(s)
                logits = model.collision_logits(forward, backward, mm)[torch.arange(len(s), device=dev), picks[:, 2]]
                total = total + F.binary_cross_entropy_with_logits(logits, target)
            return total

        gate_params = list(model.collision.parameters())
        opt = torch.optim.Adam(gate_params, lr=cfg.get("gate_lr", 1e-3))
        count = 5 if args.smoke else cfg.get("gate_updates", 30000)
        for update in range(1, count + 1):
            for group in opt.param_groups:
                group["lr"] = cfg.get("gate_lr", 1e-3) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * update / count)))
            loss = collision_loss(collision_batch(cfg.get("gate_batch", 2048)))
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
                value = collision_loss(fixed)
                value.backward()
                return value
            for _ in range(cfg["gate_lbfgs"]):
                opt.step(gate_closure)
        with torch.no_grad():
            report = {}
            for picks, (t, _, _, _) in zip(collision_batch(60000), strata):
                s = states[picks[:, 0], picks[:, 1]]
                truth_on = coll_labels[picks[:, 0], picks[:, 1], picks[:, 2]]
                forward, _, backward, _, mm = model.all_pair_features(s)
                on = model.collision_logits(forward, backward, mm)[torch.arange(len(s), device=dev), picks[:, 2]] >= 0
                report[f"type{t}"] = dict(fp=int((on & ~truth_on).sum()), fn=int((~on & truth_on).sum()), positives=int(truth_on.sum()))
            event("collision_fit", **report)
        for p in gate_params:
            p.requires_grad_(False)

        # ---- 3. contact network ----------------------------------------------
        forced = dilate(labels, reach) & ~after
        gap = cfg.get("band_gap_records", 0)
        if gap:
            # Leave out band windows that end within `gap` records before the pair's contact.
            c = contacts.to(torch.float32)
            cum = torch.cat((torch.zeros_like(c[:, :1]), c.cumsum(1)), 1)
            ahead = torch.zeros_like(labels)
            last = cum.shape[1] - 1
            idx = torch.arange(n_win, device=dev)
            lo, hi = (idx + stride).clamp_max(last), (idx + stride + gap).clamp_max(last)
            ahead = (cum[:, hi] - cum[:, lo]) > 0.5
            forced = forced & ~(ahead & ~labels)
        forced = forced | labels
        pos_windows = torch.nonzero(train & labels.any(-1))
        band_windows = torch.nonzero(train & forced.any(-1) & ~labels.any(-1))
        far_mask = train & ~dilate(labels, cfg.get("far_records", 50)).any(-1)
        far_windows = torch.nonzero(far_mask)
        far_windows = far_windows[torch.randperm(len(far_windows), device=dev, generator=rng)[:cfg.get("far_cache", 200000)]]

        def build(windows, far=False):
            parts = []
            for w in windows.split(32768):
                s, y = states[w[:, 0], w[:, 1]], states[w[:, 0], w[:, 1] + stride]
                with torch.no_grad():
                    core = model.core_delta(s, history(w, model.context))
                if far:
                    g = torch.zeros(len(w), n_pairs, dtype=torch.float64, device=dev)
                    g[torch.arange(len(w), device=dev), torch.randint(n_pairs, (len(w),), device=dev, generator=rng)] = 1.0
                else:
                    g = forced[w[:, 0], w[:, 1]].to(torch.float64)
                inv = torch.zeros(len(w), n_mov, dtype=torch.float64, device=dev)
                for p, (i, j) in enumerate(pairs):
                    inv[:, slot[i]] = torch.maximum(inv[:, slot[i]], g[:, p])
                    if j in slot:
                        inv[:, slot[j]] = torch.maximum(inv[:, slot[j]], g[:, p])
                parts.append((s, y, core, g, inv))
            return tuple(torch.cat([part[k] for part in parts]) for k in range(5))

        caches = {"pos": build(pos_windows), "band": build(band_windows) if len(band_windows) else None,
                  "far": build(far_windows, far=True) if len(far_windows) else None}
        event("contact_cache", positive_windows=len(pos_windows), band_windows=len(band_windows), far_windows=len(far_windows))
        with torch.no_grad():
            s, y, core, g, inv = caches["pos"]
            single = g.sum(-1) == 1
            s, y, core, g = s[single], y[single], core[single], g[single]
            forward, rot_f, backward, rot_b, mm = model.all_pair_features(s)
            which = g.argmax(-1)
            ar = torch.arange(len(s), device=dev)
            first = torch.tensor([slot[i] for i, _ in pairs], device=dev)[which]
            rows = [(forward[ar, which], rot_f[ar, which], (y - s - core)[ar, first], pair_type[which])]
            for k, p in enumerate(mm.tolist()):
                sel = which == p
                if sel.any():
                    rows.append((backward[ar[sel], k], rot_b[ar[sel], k], (y - s - core)[ar[sel], slot[pairs[p][1]]], pair_type[which[sel]]))
            x = torch.cat([r[0] for r in rows])
            rot = torch.cat([r[1] for r in rows])
            tgt_world = torch.cat([r[2] for r in rows])
            kind = torch.cat([r[3] for r in rows])
            tgt = torch.cat([(tgt_world[:, k:k + 3].unsqueeze(-2) @ rot).squeeze(-2) for k in (0, 3, 6)], -1)
            in_floor = torch.tensor([0.01] * 3 + [0.05] * 3 + [1.0] * 3 + [0.05] * 3 + [1.0] * 3 + [1e-3] * 3 + [1e-3] * 2,
                                    dtype=torch.float64, device=dev)
            fit_report = {}
            for t in range(TYPES):
                sel = kind == t
                if sel.sum() < 2:
                    continue
                xt, yt = x[sel], tgt[sel]
                model.in_mean2[t] = xt.mean(0)
                model.in_std2[t] = torch.maximum(xt.std(0), in_floor)
                model.out_mean2[t] = yt.mean(0)
                model.out_std2[t] = yt.std(0).clamp_min(1e-4)
                f = (xt - model.in_mean2[t]) / model.in_std2[t]
                design = torch.cat((f, torch.ones_like(f[:, :1])), -1).cpu()
                yy = ((yt - model.out_mean2[t]) / model.out_std2[t]).cpu()
                gram = design.T @ design + cfg.get("skip_ridge", 1e-3) * len(design) * torch.eye(design.shape[1], dtype=design.dtype)
                beta = torch.linalg.solve(gram, design.T @ yy).to(dev)
                model.contact_skip[t].weight.copy_(beta[:-1].T)
                model.contact_skip[t].bias.copy_(beta[-1])
                fit_report[f"type{t}"] = dict(windows=int(sel.sum()), max_skip_weight=float(beta[:-1].abs().max()))
            for key in ("compress", "gate_mean2", "gate_std2", "in_mean2", "in_std2", "out_mean2", "out_std2"):
                norm[key] = getattr(model, key).tolist()
        event("contact_linear_fit", **fit_report)

        def contact_loss(s, y, core, g, inv):
            _, d1, d2, mm = model.contact_terms(s)
            pred = s + core + model.scatter(s, g, d1, d2, mm)
            err = ((pred - y) * model.mask / response_scale).square().mean(-1)
            return (err * inv).sum() / inv.sum().clamp_min(1)

        mix = cfg.get("contact_mix", {"pos": 0.45, "band": 0.45, "far": 0.10})

        def mixture(n):
            out = []
            for name, share in mix.items():
                cache = caches.get(name)
                m = int(round(n * share))
                if cache is None or m == 0:
                    continue
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
            loss = contact_loss(*mixture(64 if args.smoke else cfg.get("contact_batch", 4096)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(contact_params, 10.0)
            opt.step()
            if update % cfg.get("evaluate_every", 10000) == 0 or update == count:
                check("contact_adam", update, loss.detach())
        if cfg.get("contact_lbfgs", 0) and not args.smoke:
            fixed = mixture(cfg.get("contact_lbfgs_batch", 200000))
            opt = torch.optim.LBFGS(contact_params, lr=1.0, max_iter=15, history_size=40, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-10, tolerance_change=1e-14)

            def contact_closure():
                opt.zero_grad()
                value = contact_loss(*fixed)
                value.backward()
                return value
            deadline = cfg.get("lbfgs_deadline_s", 9000)  # leave time for refinement inside the 4 h job limit
            for update in range(1, cfg["contact_lbfgs"] + 1):
                opt.step(contact_closure)
                if update % cfg.get("lbfgs_check_every", 25) == 0 or update == cfg["contact_lbfgs"]:
                    check("contact_lbfgs", update, contact_closure().detach())
                    if time.perf_counter() - start > deadline:
                        event("contact_lbfgs_deadline", update=update)
                        break
    else:
        check("init", 0, torch.zeros(()))

    # ---- 4. rollout refinement ---------------------------------------------
    phases = cfg.get("refine_phases", [])
    if phases and not cfg.get("init_checkpoint"):
        retention = mixture
    else:
        retention = None
    model, _ = load_any(output / "best.pt", dev)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    contact_params = list(model.contact.parameters()) + list(model.contact_skip.parameters())
    for p in contact_params:
        p.requires_grad_(True)
    contact_windows = torch.nonzero(train & labels.any(-1))
    any_window = torch.nonzero(train)
    for phase, spec in enumerate(phases):
        count = 2 if args.smoke else spec["updates"]
        horizon = 4 if args.smoke else round(spec["horizon_s"] / model.dt)
        rb = 4 if args.smoke else spec.get("batch", 256)
        opt = torch.optim.Adam(contact_params, lr=spec["lr"])
        for update in range(1, count + 1):
            starts = sample(contact_windows, rb // 2).clone()
            starts[:, 1] = (starts[:, 1] - torch.randint(1, horizon, (len(starts),), device=dev, generator=rng) * stride).clamp_min(0)
            w = torch.cat((starts, sample(any_window, rb - len(starts))))
            current = states[w[:, 0], w[:, 1]]
            hist = history(w, model.context)
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
            if retention is not None and spec.get("retention", 1.0) > 0:
                s, y, core, g, inv = retention(cfg.get("contact_batch", 4096) // 4)
                _, d1, d2, mm = model.contact_terms(s)
                pred = s + core + model.scatter(s, g, d1, d2, mm)
                err = ((pred - y) * model.mask / response_scale).square().mean(-1)
                loss = loss + spec.get("retention", 1.0) * (err * inv).sum() / inv.sum().clamp_min(1)
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
                at_target_p95_mm=1000 * report["at_target_m"]["p95"], rmse_p95_mm=1000 * report["rmse_m"]["p95"],
                rmse_median_mm=1000 * report["rmse_m"]["median"], end_p95_mm=1000 * report["end_m"]["p95"])
    atomic_json(output / "complete.json", done)
    event("complete", **done)


if __name__ == "__main__":
    main()
