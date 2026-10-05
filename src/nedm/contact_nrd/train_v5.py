"""Training of the unified contact NRD, version 5: one configuration for every system.

The same file, network design and settings train the ball and pool models; only the data differs. Every scale
comes from a data rule. Stages (each writes a checkpoint, so a later job resumes after the last finished stage):

1. Core: free steps of each moving body (log-spaced bins of the velocity change, weight ~ sqrt(count); bodies at
   rest included, no rest anchor). Ridge-fitted linear path on the current state; output scale = residual std
   (floor 1e-3 x target RMS). Phase A clean, phase B with history noise and history dropout.
2. Collision: binary cross-entropy on exact labels, strata per pair (contact / near / random); only the sampled
   pair is evaluated. History noise and dropout.
3. Contact: per-pair output scale and ridge linear path from single-contact steps; MSE on the next state with the
   true switches, loss scale kappa x RMS of the contact change per channel. History noise, dropout, and
   pushforward: a share of the samples gets the model's own history (m no-gradient model steps from t - m).
4. Rollout refinement (Huber on free rollouts + teacher-forced retention).

Noise: i.i.d. Gaussian on every history difference, sigma = noise_scale x RMS of the core's own one-step error
per channel (measured after core phase A). History dropout: the usable history length is uniform in 1..K.
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
from nedm.contact_nrd.model_v5 import STATE, UnifiedNRD, load_v5
from nedm.contact_nrd.train import atomic_json, dilate

POS, VEL, SPIN = slice(0, 3), slice(3, 6), slice(6, 9)


def cosine(update, count, warm, floor):
    return min(1.0, update / warm if warm else 1.0) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * update / count)))


def ridge(design, target, lam):
    gram = design.T @ design + lam * len(design) * torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
    return torch.linalg.solve(gram, design.T @ target)


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
    smoke = args.smoke
    data = load_data(args.data, dev, max_episodes=cfg.get("max_episodes"), model_step_s=cfg.get("model_step_s"))
    system = data["system"]
    states, splits, lengths, stride, contacts = data["states"], data["splits"], data["lengths"], data["stride"], data["contacts"]
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
    touched = labels | after
    involved = torch.zeros(n_ep, n_win, n_mov, dtype=torch.bool, device=dev)
    for p, (i, j) in enumerate(pairs):
        involved[..., slot[i]] |= touched[..., p]
        if j in slot:
            involved[..., slot[j]] |= touched[..., p]
    val = torch.nonzero(splits == 1).flatten()
    if smoke:
        val = val[:24]
    rng = torch.Generator(device=dev).manual_seed(cfg["seed"] + 1000)
    K = int(cfg["history"])

    def sample(pool, n):
        return pool[torch.randint(len(pool), (n,), device=dev, generator=rng)]

    def hist_of(windows):
        """windows [n, 2] (episode, record) -> true history [n, K, D, 9] (zeros where missing), ok [n, K]."""
        raw = windows[:, 1, None] + torch.arange(1 - K, 1, device=dev) * stride
        ok = raw >= 0
        h = states[windows[:, 0, None], raw.clamp_min(0)] * ok[..., None, None]
        return h, ok

    def drop_history(ok):
        keep = torch.randint(1, K + 1, (len(ok),), device=dev, generator=rng)
        return ok & (torch.arange(K, device=dev)[None, :] >= K - keep[:, None])

    # ---- state mask (constant channels: v2 data rule) ------------------------------
    sw = sample(torch.nonzero(train), 100000)
    flat0 = states[sw[:, 0], sw[:, 1]].reshape(-1, STATE)
    span = flat0.quantile(0.999, dim=0) - flat0.quantile(0.001, dim=0)
    mask = torch.ones(STATE, dtype=torch.float64, device=dev)
    for cols in ((0, 1, 2), (3, 4, 5), (6, 7, 8)):
        biggest = span[list(cols)].max()
        for c in cols:
            if span[c] < 0.01 * biggest:
                mask[c] = 0.0

    mc = {"architecture": "contact_graph_nrd_v5", "dt_s": data["dt"], "history": K, "network": cfg["network"],
          "trunk_dtype": cfg.get("trunk_dtype", "float32")}
    model = UnifiedNRD(mc, system, mask.tolist()).to(dev)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sigma = None

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_s=round(time.perf_counter() - start, 1), **extra)
        with (output / "train_log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    runtime = dict(config=cfg, model_config=mc, system=system["name"], training_episodes=int((splits == 0).sum()),
                   validation_episodes=len(val), test_used_for_selection=False, host=platform.node(),
                   job_id=os.environ["SLURM_JOB_ID"], torch=torch.__version__, mask=mask.tolist(), stride=stride, dt_s=data["dt"],
                   parameters={n: sum(p.numel() for p in getattr(model, n).parameters()) for n in ("core", "collision", "contact")},
                   data_sha256=data["index"]["data_sha256"],
                   source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")})
    if not (output / "run_config.json").exists():
        atomic_json(output / "run_config.json", runtime)
    else:
        event("resume", job_id=os.environ["SLURM_JOB_ID"])

    def save(path, stage, update, validation=None):
        torch.save(dict(model_config=mc, system=system, mask=mask.tolist(), model_state_dict=model.state_dict(), runtime=runtime,
                        stage=stage, update=update, validation=validation, sigma=None if sigma is None else sigma.tolist()), path)

    def load_stage(path):
        nonlocal sigma
        packet = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(packet["model_state_dict"])
        sigma = None if packet.get("sigma") is None else torch.tensor(packet["sigma"], dtype=torch.float64, device=dev)
        return packet

    def noise():
        return None if (sigma is None or K == 1) else sigma * cfg.get("noise_scale", 1.0)

    best = {"score": float("inf")}
    checks = {"count": 0}

    def check(stage, update, loss, extra=None):
        was = model.training
        model.eval()
        with torch.no_grad():
            report = free_metrics(model, data, val)
        checks["count"] += 1
        if not math.isfinite(report["selection_score"]):
            report["selection_score"] = 1e9
        if report["selection_score"] < best["score"] or not (output / "best.pt").exists():
            best["score"] = report["selection_score"]
            save(output / "best.pt", stage, update, report)
        save(output / "last.pt", stage, update, report)
        event(stage, update=update, loss=float(loss), rmse_median_mm=1000 * report["rmse_m"]["median"],
              rmse_p95_mm=1000 * report["rmse_m"]["p95"], at_target_median_mm=1000 * report["at_target_m"]["median"],
              at_target_p95_mm=1000 * report["at_target_m"]["p95"], eligible_p95_mm=1000 * report["eligible_at_target_m"]["p95"],
              end_p95_mm=1000 * report["end_m"]["p95"], events_ok=report["event_ok_fraction"], score=report["selection_score"],
              **(extra or {}))
        if was:
            model.train()

    # ---- 1. core -------------------------------------------------------------------
    core = model.core
    if (output / "core.pt").exists():
        load_stage(output / "core.pt")
        event("core_loaded")
    else:
        phase_keep = (torch.arange(n_win, device=dev) % cfg.get("free_phase_step", 5) == 0)[None, :]
        triples = []
        for b in range(n_mov):
            w = torch.nonzero(train & phase_keep & ~involved[..., b])
            triples.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
        triples = torch.cat(triples)
        with torch.no_grad():
            dvel = torch.cat([(states[c[:, 0], c[:, 1] + stride, c[:, 2], 3:6] - states[c[:, 0], c[:, 1], c[:, 2], 3:6]).norm(dim=-1)
                              for c in triples.split(1 << 20)])
            logd = torch.log10(dvel.clamp_min(1e-12))
            sub = logd[torch.randperm(len(logd), device=dev)[:1000000]].float()
            lo, hi = torch.quantile(sub, 0.001).double(), torch.quantile(sub, 0.999).double()
            bins = cfg.get("core_bins", 8)
            which = torch.zeros_like(logd, dtype=torch.long) if hi - lo < 1e-6 else ((logd - lo) / (hi - lo) * bins).floor().clamp(0, bins - 1).long()
            groups = [triples[which == k] for k in range(bins) if (which == k).any()]
            weights = torch.tensor([math.sqrt(len(g)) for g in groups], dtype=torch.float64)
            weights = (weights / weights.sum()).tolist()

        def core_batch(n):
            tr = torch.cat([sample(g, max(1, int(round(n * w)))) for g, w in zip(groups, weights)])
            h, ok = hist_of(tr[:, :2])
            h = h[torch.arange(len(tr), device=dev), :, tr[:, 2]][:, :, None]                 # [n, K, 1, 9]
            target = (states[tr[:, 0], tr[:, 1] + stride, tr[:, 2]] - h[:, -1, 0]) * mask
            return h, ok, target

        def core_pred(h, ok, nz=None):
            cur, diff, lag_ok = model.code(h, ok, nz)
            return model.run_core(cur, diff, lag_ok)[:, 0]

        with torch.no_grad():
            h, ok, target = core_batch(128 if smoke else 200000)
            cur, diff = h[:, -1, 0], h[:, 1:, 0] - h[:, :-1, 0]
            fl = torch.tensor([0.01] * 6 + [0.1] * 3, dtype=torch.float64, device=dev)
            dfl = torch.tensor([1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3, dtype=torch.float64, device=dev)
            core.in_mean[0, 0] = cur.mean(0)
            core.in_std[0, 0] = torch.maximum(cur.std(0), fl)
            dv = diff[ok[:, :-1]] if K > 1 else torch.zeros(2, STATE, dtype=torch.float64, device=dev)
            core.diff_std[0, 0] = torch.maximum(dv.std(0), dfl)
            zc = (cur - core.in_mean[0, 0]) / core.in_std[0, 0] * mask
            design = torch.cat((zc, torch.ones_like(zc[:, :1])), -1)
            mean = target.mean(0)
            beta = ridge(design, target - mean, cfg.get("ridge", 1e-6))
            resid = target - mean - design @ beta
            rms = target.square().mean(0).sqrt()
            core.out_mean[0] = mean
            # Output scale: residual std after the linear path, floored at 1e-3 x the target RMS and at 1e-4 x the
            # state scale of the channel (a channel whose free change is exactly zero, e.g. spin in flight).
            core.out_std[0] = torch.maximum(torch.maximum(resid.std(0), 1e-3 * rms), 1e-4 * core.in_std[0, 0])
            core.lin.data[0] = beta / core.out_std[0]
        event("core_linear_fit", residual_over_rms=(resid.std(0) / rms.clamp_min(1e-30)).tolist(), bins=[len(g) for g in groups])
        with torch.no_grad():
            pv = []
            for b in range(n_mov):
                w = torch.nonzero(vwin & ~involved[..., b])
                w = w[torch.randperm(len(w), device=dev, generator=rng)[:100000]]
                pv.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
            pv = torch.cat(pv)
            pv_h, pv_ok = hist_of(pv[:, :2])
            pv_h = pv_h[torch.arange(len(pv), device=dev), :, pv[:, 2]][:, :, None]
            pv_t = (states[pv[:, 0], pv[:, 1] + stride, pv[:, 2]] - pv_h[:, -1, 0]) * mask
            pv_moving = pv_h[:, -1, 0, 3:9].abs().amax(-1) > 1e-6

        def core_val():
            with torch.no_grad():
                err = torch.cat([core_pred(h, o) for h, o in zip(pv_h.split(16384), pv_ok.split(16384))]) - pv_t
            loss = float((err / core.out_std[0]).square()[:, mask > 0].mean())
            return err, loss

        params = list(core.parameters())
        for spec_name, noisy in (("core_clean", False), ("core_noisy", True)):
            spec = cfg[spec_name]
            if noisy:
                err, _ = core_val()
                sigma = err[pv_moving].square().mean(0).sqrt() * mask
                event("core_sigma", sigma=sigma.tolist())
            opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=0.0)
            count = 5 if smoke else spec["updates"]
            for update in range(1, count + 1):
                for g in opt.param_groups:
                    g["lr"] = spec["lr"] * cosine(update, count, spec.get("warmup", 0), 0.01)
                h, ok, target = core_batch(64 if smoke else spec["batch"])
                if noisy:
                    ok = drop_history(ok)
                pred = core_pred(h, ok, noise() if noisy else None)
                loss = ((pred - target) / core.out_std[0]).square()[:, mask > 0].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if update % 10000 == 0:
                    event(spec_name, update=update, loss=float(loss))
            err, vloss = core_val()
            event(f"{spec_name}_val", val_loss=vloss)
        with torch.no_grad():
            report = {name: {q: float(torch.quantile(err[pv_moving][:, cols].norm(dim=-1), qq)) for q, qq in (("median", 0.5), ("p95", 0.95))}
                      for name, cols in (("pos_m", [0, 1, 2]), ("vel_mps", [3, 4, 5]), ("spin_radps", [6, 7, 8]))}
            rest = pv_h[:, -1:, 0].clone()
            rest[..., 3:9] = 0.0
            rest_h = rest[:, :, None].expand(-1, K, -1, -1).clone()
            creep = torch.cat([core_pred(h, torch.ones(len(h), K, dtype=torch.bool, device=dev)) for h in rest_h.split(16384)])
            report["rest_creep_mps_max"] = float(creep[:, 3:6].norm(dim=-1).max())
            report["rest_creep_mps_median"] = float(creep[:, 3:6].norm(dim=-1).median())
            if K > 1:
                hh = pv_h.clone()
                hh[:, :-1] = hh[:, :-1] + torch.randn_like(hh[:, :-1]) * sigma
                d = torch.cat([core_pred(h, o) for h, o in zip(hh.split(16384), pv_ok.split(16384))]) - (err + pv_t)
                report["older_noise_response_over_sigma"] = float((d[:, 3:6].norm(dim=-1) / sigma[3:6].norm().clamp_min(1e-30)).median())
        atomic_json(output / "core_validation.json", report)
        event("core_validation", **{k: (v["p95"] if isinstance(v, dict) else v) for k, v in report.items()})
        save(output / "core.pt", "core", 0)
    for p in core.parameters():
        p.requires_grad_(False)
    if sigma is None:
        raise RuntimeError("core noise scale missing")

    # ---- 2. collision -------------------------------------------------------------
    coll = model.collision
    near_rec = cfg.get("near_negative_records", 50)
    near_pair = dilate(labels, near_rec) & train[..., None]

    def pair_raw(windows, pairs_, hist=None):
        """Role-ordered raw inputs of the pair networks for windows [n, 2] and pairs [n]."""
        h, ok = hist_of(windows) if hist is None else hist
        cur, diff, lag_ok = model.code(h, ok)
        rows = torch.arange(len(windows), device=dev)
        return model.pair_inputs(cur, diff, lag_ok, rows, pairs_)

    def fit_inputs(net, pools):
        """Per-pair input standardisation from the given window pools (list of [n, 2] per pair)."""
        fl = torch.tensor([0.01] * 6 + [0.1] * 3, dtype=torch.float64, device=dev)
        dfl = torch.tensor([1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3, dtype=torch.float64, device=dev)
        with torch.no_grad():
            for p, wins in enumerate(pools):
                wins = wins[torch.randperm(len(wins), device=dev, generator=rng)[:50000]]
                c, df, present, lag_ok, _, _ = pair_raw(wins, torch.full((len(wins),), p, device=dev))
                for q in range(c.shape[1]):
                    if not present[:, q].any():
                        continue
                    x = c[present[:, q], q]
                    net.in_mean[p, q] = x.mean(0)
                    net.in_std[p, q] = torch.maximum(x.std(0), fl)
                    if K > 1:
                        okd = lag_ok[:, 1:] & present[:, q, None]
                        net.diff_std[p, q] = torch.maximum(df[:, :, q][okd].std(0), dfl)

    any_train = torch.nonzero(train)
    pools = []
    for p in range(n_pairs):
        w = torch.nonzero(near_pair[..., p])
        pools.append(w if len(w) > 1000 else torch.cat((w, sample(any_train, 50000))))
    near = dilate(labels, near_rec) & ~labels & ~after
    strata = [(torch.nonzero(labels[..., p] & train), torch.nonzero(near[..., p] & train),
               torch.nonzero(labels[..., p] & vwin), torch.nonzero(near[..., p] & vwin)) for p in range(n_pairs)]
    any_val = torch.nonzero(vwin)

    def collision_batch(n, split="train"):
        out = []
        for p, (pos, neg, vpos, vneg) in enumerate(strata):
            pos, neg, anyw = (pos, neg, any_train) if split == "train" else (vpos, vneg, any_val)
            m = max(3, n // n_pairs)
            parts = [sample(pos, m // 3)] if len(pos) else []
            parts += [sample(neg, m // 3)] if len(neg) else []
            parts.append(sample(anyw, m - sum(len(q) for q in parts)))
            wnd = torch.cat(parts)
            out.append(torch.cat((wnd, torch.full_like(wnd[:, :1], p)), -1))
        return torch.cat(out)

    def collision_logits(picks, train_mode):
        h, ok = hist_of(picks[:, :2])
        if train_mode:
            ok = drop_history(ok)
        cur, diff, lag_ok = model.code(h, ok, noise() if train_mode else None)
        return model.logits(cur, diff, lag_ok, torch.arange(len(picks), device=dev), picks[:, 2])

    def collision_loss(picks, train_mode=False):
        total = 0.0
        for c in picks.split(65536):
            target = labels[c[:, 0], c[:, 1], c[:, 2]].to(torch.float64)
            total = total + F.binary_cross_entropy_with_logits(collision_logits(c, train_mode), target, reduction="sum")
        return total / len(picks)

    if (output / "collision.pt").exists():
        load_stage(output / "collision.pt")
        event("collision_loaded")
    else:
        fit_inputs(coll, pools)
        spec = cfg["collision"]
        params = [p for p in coll.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=0.0)
        count = 5 if smoke else spec["updates"]
        for update in range(1, count + 1):
            for g in opt.param_groups:
                g["lr"] = spec["lr"] * cosine(update, count, spec.get("warmup", 0), 0.01)
            loss = collision_loss(collision_batch(spec["batch"]), True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if update % 10000 == 0:
                event("collision_adam", update=update, loss=float(loss))
        with torch.no_grad():
            val_picks = collision_batch(60000, "val")
            report = {"bce": float(collision_loss(val_picks))}
            for p in range(n_pairs):
                sel = val_picks[val_picks[:, 2] == p]
                on = collision_logits(sel, False) >= 0
                truth = labels[sel[:, 0], sel[:, 1], p]
                report[f"pair{p}"] = dict(fp=int((on & ~truth).sum()), fn=int((~on & truth).sum()), positives=int(truth.sum()))
        event("collision_fit", **report)
        save(output / "collision.pt", "collision", 0)
    for p in coll.parameters():
        p.requires_grad_(False)

    # ---- 3. contact ---------------------------------------------------------------
    cont = model.contact
    pos_windows = torch.nonzero(train & labels.any(-1))

    def own_history(windows, m):
        """The model's own history at windows [n, 2]: m [n] no-gradient model steps from t - m*stride (true
        history before that), with the true switches at every step (so a switch timing error never enters).
        Rows with t - m*stride < 0 keep the true history."""
        t0 = windows[:, 1] - m * stride
        okm = t0 >= 0
        m = torch.where(okm, m, torch.zeros_like(m))
        t0 = torch.where(okm, t0, windows[:, 1])
        h, ok = hist_of(torch.stack((windows[:, 0], t0), -1))
        with torch.no_grad():
            for step in range(int(m.max()) if len(m) else 0):
                act = step < m
                if not act.any():
                    break
                idx = torch.nonzero(act).flatten()
                g = labels[windows[idx, 0], t0[idx] + step * stride].to(torch.float64)
                nxt = model.step_with_gates(h[idx], ok[idx], g)
                h2, ok2 = model.advance(h[idx], ok[idx], nxt)
                h, ok = h.clone(), ok.clone()
                h[idx], ok[idx] = h2, ok2
        return h, ok

    push_stats = {"tried": 0, "kept": 0}

    def contact_example(windows, train_mode, push=0.0):
        h, ok = hist_of(windows)
        if push > 0:
            sel = torch.nonzero(torch.rand(len(windows), device=dev, generator=rng) < push).flatten()
            if len(sel):
                m = torch.randint(1, cfg.get("push_max_steps", 3) + 1, (len(sel),), device=dev, generator=rng)
                hs, oks = own_history(windows[sel], m)
                # Keep an own history only if its current state is within push_max_dev loss scales of the truth
                # (the target is the true next state, which is consistent only for small deviations).
                dev_ = ((hs[:, -1] - h[sel, -1]) * mask / scale).abs().amax((-1, -2))
                keep = dev_ <= cfg.get("push_max_dev", 3.0)
                push_stats["tried"] += len(sel)
                push_stats["kept"] += int(keep.sum())
                sel, hs, oks = sel[keep], hs[keep], oks[keep]
                h, ok = h.clone(), ok.clone()
                h[sel], ok[sel] = hs, oks
        if train_mode:
            ok = drop_history(ok)
        g = labels[windows[:, 0], windows[:, 1]].to(torch.float64)
        pred = model.step_with_gates(h, ok, g, noise() if train_mode else None)
        y = states[windows[:, 0], windows[:, 1] + stride]
        affected = torch.zeros(len(windows), n_mov, dtype=torch.float64, device=dev)
        for p in range(n_pairs):
            on = g[:, p] > 0.5
            affected[on, model.pair_pos[p, 0]] = 1.0
            if model.pair_mm[p]:
                affected[on, model.pair_pos[p, 1]] = 1.0
        return pred, y, affected, h[:, -1]

    resume_path = output / "contact_state.pt"
    if (output / "contact_done.json").exists():
        best["score"] = json.loads((output / "contact_done.json").read_text())["best"]
        event("contact_loaded", best=best["score"])
    elif resume_path.exists():
        event("contact_resume_found")
    else:
        with torch.no_grad():
            cpools = []
            for p in range(n_pairs):
                w = torch.nonzero(labels[..., p] & train)
                cpools.append(w if len(w) > 200 else pools[p])
            fit_inputs(cont, cpools)
            # Output scale and linear path per pair, from steps where only that pair is in contact.
            g = labels[pos_windows[:, 0], pos_windows[:, 1]]
            single = pos_windows[g.sum(-1) == 1]
            single = single[torch.randperm(len(single), device=dev, generator=rng)[:400000]]
            sp = labels[single[:, 0], single[:, 1]].float().argmax(-1)
            h, ok = hist_of(single)
            cur, diff, lag_ok = model.code(h, ok)
            core_d = torch.cat([model.run_core(*t) for t in zip(cur.split(65536), diff.split(65536), lag_ok.split(65536))])
            resid = (states[single[:, 0], single[:, 1] + stride] - cur - core_d) * mask       # [n, D, 9]
            c, df, present, lo, fx, q = model.pair_inputs(cur, diff, lag_ok, torch.arange(len(single), device=dev), sp)
            zc, _ = cont.scaled(c, df, present, lo, q, mask)
            design = torch.cat((zc.flatten(1), torch.ones_like(zc[:, :1, 0])), -1)
            part2 = torch.where(model.pair_mm[sp][:, None], resid[torch.arange(len(sp), device=dev), model.pair_pos[sp, 1].clamp_min(0)],
                                torch.zeros_like(resid[:, 0]))
            tgt = torch.cat((resid[torch.arange(len(sp), device=dev), model.pair_pos[sp, 0]], part2), -1)   # [n, 18]
            fit_report = {}
            present_pairs = []
            for p in range(n_pairs):
                sel = sp == p
                if sel.sum() < 20:
                    continue
                present_pairs.append(p)
                x, y = design[sel], tgt[sel]
                mean = y.mean(0)
                beta = ridge(x, y - mean, cfg.get("skip_ridge", 1e-3))
                r = y - mean - x @ beta
                rms = y.square().mean(0).sqrt()
                cont.out_mean[p] = mean
                cont.out_std[p] = torch.maximum(torch.maximum(r.std(0), 1e-3 * rms), 1e-4 * cont.in_std[p, 0].repeat(2))
                cont.lin.data[p] = beta / cont.out_std[p]
                fit_report[f"pair{p}"] = dict(steps=int(sel.sum()), linear_r2=float(1 - r.square().sum() / (y - mean).square().sum().clamp_min(1e-30)))
            for p in range(n_pairs):
                if p not in present_pairs:
                    cont.out_std[p] = cont.out_std[present_pairs].mean(0)
            # Loss scale per channel: kappa x RMS of the contact change of the affected bodies.
            aff = torch.cat((resid[torch.arange(len(sp), device=dev), model.pair_pos[sp, 0]],
                             resid[torch.arange(len(sp), device=dev), model.pair_pos[sp, 1].clamp_min(0)][model.pair_mm[sp]]))
            rms = aff.square().mean(0).sqrt()
            kappa = torch.tensor([cfg["kappa"][0]] * 3 + [cfg["kappa"][1]] * 3 + [cfg["kappa"][2]] * 3, dtype=torch.float64, device=dev)
            scale = torch.where(mask > 0, (kappa * rms).clamp_min(1e-12), torch.ones_like(rms))
            norm_scale = scale.tolist()
        event("contact_linear_fit", loss_scale=norm_scale, **fit_report)
        (output / "contact_scale.json").write_text(json.dumps({"scale": norm_scale}))

    scale = torch.tensor(json.loads((output / "contact_scale.json").read_text())["scale"], dtype=torch.float64, device=dev)

    # Fixed validation contact steps for one-step metrics (true history and the model's own history).
    vpos = torch.nonzero(vwin & labels.any(-1))
    vpos = vpos[torch.randperm(len(vpos), device=dev, generator=rng)[: (200 if smoke else 20000)]]

    def onestep():
        with torch.no_grad():
            out = {}
            for name, push in (("true", 0.0), ("own", 1.0)):
                errs = []
                for c in vpos.split(4096):
                    pred, y, aff, _ = contact_example(c, False, push)
                    e = (pred - y)[..., 3:6].norm(dim=-1)                                      # [n, D] m/s
                    errs.append(e[aff > 0])
                e = torch.cat(errs)
                out[f"onestep_{name}_vel_median_mps"] = float(e.median())
                out[f"onestep_{name}_vel_p95_mps"] = float(torch.quantile(e.float(), 0.95))
            return out

    def contact_loss(windows, push, train_mode=True):
        pred, y, aff, _ = contact_example(windows, train_mode, push)
        err = ((pred - y) * mask / scale).square().mean(-1)
        return (err * aff).sum() / aff.sum().clamp_min(1)

    params = [p for p in cont.parameters() if p.requires_grad]
    if not (output / "contact_done.json").exists():
        spec = cfg["contact"]
        opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=0.0)
        count = 5 if smoke else spec["updates"]
        push_from = int(cfg.get("push_start", 0.25) * count)
        first = 1
        if resume_path.exists():
            state = torch.load(resume_path, map_location=dev, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            opt.load_state_dict(state["optimizer"])
            rng.set_state(state["rng"])
            best["score"], first = state["best"], state["update"] + 1
            event("contact_resumed", update=state["update"], best=best["score"])
        for update in range(first, count + 1):
            for g in opt.param_groups:
                g["lr"] = spec["lr"] * cosine(update, count, spec.get("warmup", 0), 0.01)
            push = cfg.get("push_prob", 0.5) if update > push_from else 0.0
            loss = contact_loss(sample(pos_windows, 64 if smoke else spec["batch"]), push)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if update % (5 if smoke else cfg.get("evaluate_every", 10000)) == 0 or update == count:
                check("contact_adam", update, loss.detach(), {**onestep(), "push_kept_fraction": push_stats["kept"] / max(1, push_stats["tried"])})
                push_stats.update(tried=0, kept=0)
                tmp = output / "contact_state.pt.tmp"
                torch.save(dict(model_state_dict=model.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                                best=best["score"], update=update), tmp)
                tmp.replace(resume_path)
        (output / "contact_done.json").write_text(json.dumps({"best": best["score"]}))

    # ---- 4. rollout refinement -----------------------------------------------------
    if not (output / "complete.json").exists():
        load_stage(output / "best.pt")
        model.train()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        any_window = torch.nonzero(train)
        rscale = scale * cfg.get("rollout_scale_factor", 4.0)
        for phase, spec in enumerate(cfg.get("refine_phases", [])):
            count = 2 if smoke else spec["updates"]
            horizon = 4 if smoke else max(1, round(spec["horizon_s"] / model.dt))
            rb = 4 if smoke else spec.get("batch", 256)
            opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=0.0)
            for update in range(1, count + 1):
                starts = sample(pos_windows, rb // 2).clone()
                starts[:, 1] = (starts[:, 1] - torch.randint(1, horizon, (len(starts),), device=dev, generator=rng) * stride).clamp_min(0)
                w = torch.cat((starts, sample(any_window, rb - len(starts))))
                h, ok = hist_of(w)
                current = h[:, -1]
                total, den = current.sum() * 0.0, 0
                for step in range(horizon):
                    j = w[:, 1] + (step + 1) * stride
                    active = j < lengths[w[:, 0]]
                    y = states[w[:, 0], j.clamp_max(n_rec - 1)]
                    pred = model(h, ok)
                    err = F.huber_loss((pred - y) * mask / rscale, torch.zeros_like(pred), reduction="none", delta=1.0)
                    total = total + (err.mean((-1, -2)) * active).sum()
                    den += int(active.sum())
                    current = torch.where(active[:, None, None], pred, current)
                    h, ok = model.advance(h, ok, current)
                loss = total / max(1, den)
                if spec.get("retention", 1.0) > 0:
                    loss = loss + spec.get("retention", 1.0) * contact_loss(sample(pos_windows, cfg["contact"]["batch"] // 4), 0.0)
                opt.zero_grad()
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(gnorm):
                    raise RuntimeError("nonfinite rollout gradient")
                opt.step()
                if update % (2 if smoke else spec.get("check_every", 100)) == 0 or update == count:
                    check(f"rollout{phase}", update, loss.detach())
        model, metadata = load_v5(output / "best.pt", dev)
        with torch.no_grad():
            report = free_metrics(model, data, val)
        atomic_json(output / "validation.json", report)
        done = dict(complete=True, smoke=smoke, test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                    checkpoint_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(), best_stage=metadata["stage"],
                    best_update=metadata["update"], validation_score=report["selection_score"], checkpoints_compared=checks["count"],
                    at_target_median_mm=1000 * report["at_target_m"]["median"], at_target_p95_mm=1000 * report["at_target_m"]["p95"],
                    eligible_p95_mm=1000 * report["eligible_at_target_m"]["p95"], rmse_p95_mm=1000 * report["rmse_m"]["p95"],
                    rmse_median_mm=1000 * report["rmse_m"]["median"], end_p95_mm=1000 * report["end_m"]["p95"],
                    events_ok=report["event_ok_fraction"])
        atomic_json(output / "complete.json", done)
        event("complete", **done)


if __name__ == "__main__":
    main()
