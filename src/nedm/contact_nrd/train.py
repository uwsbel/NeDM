"""Train the contact NRD (core, collision and contact networks) on one data directory.

From the repo root (after ``python -m nedm.contact_nrd.download --part train``):

    PYTHONPATH=src python -m nedm.contact_nrd.train --data artifacts/contact_nrd/pool/train \
        --output artifacts/contact_nrd/runs/pool_s61
    PYTHONPATH=src python -m nedm.contact_nrd.train --data artifacts/contact_nrd/pool/train \
        --output artifacts/contact_nrd/runs/pool_s62 --seed 62

One configuration (configs/contact_nrd/train.json) serves every system; only the data differs. Stages:
1. core: per body kind, a ridge fit of the one-step change on contact-free steps (the core's linear path); the
   validation error of the core sets the history-noise scale (measured again at every check).
2. collision network: binary cross-entropy on contact, near-contact and any steps (history noise and dropout).
3. co-training of the core and the contact network: core loss + one-step contact loss (recorded contact switches,
   history noise and dropout; after ``push_start`` of the updates, part of the histories are the model's own
   1-3 step predictions) + free-rollout loss (horizon 0.1 -> 1 s), each term divided by its running mean.
   Every ``evaluate_every`` updates: free rollouts of ``check_episodes`` validation episodes; the best score is
   kept as best.pt (the test split is never used).
4. best.pt on every validation episode -> validation.json and complete.json.

States are velocity-level: the rate channels are step-average rates (data.step_average_rates) and the pose of
each integrated body follows pose + step x rate (model.ContactNRD.assemble). The one-step losses train only the
rate channels; the rollout loss compares the whole state.

Output directory: run_config.json, train_log.jsonl, core.pt, core_validation.json, collision.pt,
contact_scale.json, contact_state.pt (resume state), best.pt, last.pt, contact_done.json, validation.json,
complete.json. Run the same command again to resume an interrupted run: finished stages are loaded and
co-training continues from the last check. A resumed run uses the same steps, but its random draws are different,
so its result is not bit-identical to an uninterrupted run.

A full run takes 4.5-6 h on one MI350X (longer than a 4 h job limit: plan one resume). Peak GPU memory is about 12 GB (ball), 32 GB (pool) and 24 GB (arm):
the whole data set is held on the GPU in float64. Two runs with the same seed give the same result only with
--deterministic on the same machine (the embedding gradients use atomic adds on the GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from nedm.contact_nrd.data import (atomic_json, dilate, file_sha256, free_metrics, load_data, step_average_rates,
                                   window_labels)
from nedm.contact_nrd.model import ContactNRD, load_model, model_config

REPO_ROOT = Path(__file__).resolve().parents[3]
# Data rules by channel type: (state std floor, difference std floor, contact-loss kappa).
TYPE_RULES = {
    "position": (0.01, 1e-4, 1e-2), "angle": (0.01, 1e-4, 1e-2), "quaternion": (0.01, 1e-4, 1e-2),
    "velocity": (0.01, 1e-3, 5e-4), "angular_rate": (0.1, 1e-2, 5e-4), "angular_velocity": (0.1, 1e-2, 3e-3),
}


def cosine(update, count, warm, floor):
    return min(1.0, update / warm if warm else 1.0) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * update / count)))


def ridge(design, target, lam):
    gram = design.T @ design + lam * len(design) * torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
    return torch.linalg.solve(gram, design.T @ target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="data directory (system.json, unified_data.npz)")
    parser.add_argument("--output", type=Path, required=True, help="run directory (an existing run resumes)")
    parser.add_argument("--seed", type=int, help="overrides the config seed")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/contact_nrd/train.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="a few updates of every stage (pipeline check)")
    parser.add_argument("--deterministic", action="store_true", help="repeatable GPU runs on one machine (slower)")
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = json.loads(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if int(cfg["history"]) < 2:
        raise ValueError("history must be >= 2")
    job = {"job_id": os.environ["SLURM_JOB_ID"]} if os.environ.get("SLURM_JOB_ID") else {}
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    dev, smoke = args.device, args.smoke
    data = load_data(args.data, dev, model_step_s=cfg["model_step_s"])
    data = step_average_rates(data, set(cfg["integrate_kinds"]))
    system = data["system"]
    states, actions, splits, lengths, stride, contacts = (data["states"], data["actions"], data["splits"], data["lengths"],
                                                          data["stride"], data["contacts"])
    n_ep, n_rec, n_mov, smax = states.shape
    pairs = system["pairs"]
    n_pairs = len(pairs)
    moving_b = [b for b in system["bodies"] if b["moving"]]
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    slot = {k: n for n, k in enumerate(moving)}
    types = [b["channel_types"] for b in moving_b]
    labels = window_labels(contacts, stride)                                           # [N, W, P]
    # a step-average rate spans the previous model step: a contact in (r - s, r] still changes window r
    spread = torch.zeros_like(labels)
    spread[:, stride:] = labels[:, :-stride]
    labels = labels | spread
    n_win = labels.shape[1]
    previous = torch.cat((torch.zeros_like(labels[:, :1]), window_labels(contacts, 1)[:, : n_win - 1]), 1)
    after = previous & ~labels
    # In systems with actions, windows start on a command boundary (the command changes every model step);
    # systems without actions use every record.
    acting = any(int(b.get("action_dim", 0)) for b in moving_b)
    grid = ((torch.arange(n_win, device=dev) % stride == 0) if acting else torch.ones(n_win, dtype=torch.bool, device=dev))[None, :]
    valid = (torch.arange(n_win, device=dev)[None, :] < (lengths - stride)[:, None]) & grid
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
    # Checks during training use an evenly spaced subset of the validation episodes; the final report uses all.
    n_check = min(len(val), int(cfg["check_episodes"]))
    val_check = val[torch.linspace(0, len(val) - 1, n_check, device=dev).round().long()]
    rng = torch.Generator(device=dev).manual_seed(cfg["seed"] + 1000)
    K = int(cfg["history"])

    def per_channel(rule):
        """[D, S] tensor of a TYPE_RULES column for each body's channels (padding: 1)."""
        out = torch.ones(n_mov, smax, dtype=torch.float64, device=dev)
        for d, ts in enumerate(types):
            for c, t in enumerate(ts):
                out[d, c] = TYPE_RULES[t][rule]
        return out
    floor_state, floor_diff, kappa = per_channel(0), per_channel(1), per_channel(2)

    def sample(pool, n):
        return pool[torch.randint(len(pool), (n,), device=dev, generator=rng)]

    def hist_of(windows):
        raw = windows[:, 1, None] + torch.arange(1 - K, 1, device=dev) * stride
        ok = raw >= 0
        h = states[windows[:, 0, None], raw.clamp_min(0)] * ok[..., None, None]
        return h, ok

    def act_of(windows):
        return actions[windows[:, 0], windows[:, 1]]

    def drop_history(ok):
        # step-average rates: a one-entry history occurs only at an episode start (recorded rates), never later
        keep = torch.randint(2, K + 1, (len(ok),), device=dev, generator=rng)
        return ok & (torch.arange(K, device=dev)[None, :] >= K - keep[:, None])

    # ---- channel masks (constant channels by type, per body; padding off) --------------
    sw = sample(torch.nonzero(valid & (splits == 0)[:, None]), 100000)
    s0 = states[sw[:, 0], sw[:, 1]]                                                    # [n, D, S]
    span = s0.quantile(0.999, dim=0) - s0.quantile(0.001, dim=0)                       # [D, S]
    masks = torch.zeros(n_mov, smax, dtype=torch.float64, device=dev)
    for d, ts in enumerate(types):
        for c, t in enumerate(ts):
            same = [cc for cc, tt in enumerate(ts) if tt == t]
            masks[d, c] = float(span[d, c] >= 0.01 * span[d, same].max())

    mc = model_config(data["dt"], K, cfg["network"], cfg["integrate_kinds"])
    model = ContactNRD(mc, system, masks.tolist()).to(dev)
    # Channels that are trained (core and contact losses): integrated pose channels are computed, not predicted.
    core_masks = masks.clone()
    for b in torch.nonzero(model.integ_body).flatten().tolist():
        for k in (0, 1):
            cols = model.chan[b, k][model.chan[b, k] >= 0]
            core_masks[b, cols] = 0.0

    def pose_noise(sig):
        """History noise of integrated pose channels from the noise of their rates: a rate error s changes the next
        pose step by dt s (quaternion: 0.5 dt s); the core's pose outputs are not trained and do not give this noise."""
        sig = sig.clone()
        for n in torch.nonzero(model.integ_body).flatten().tolist():
            pc, qc, vc, wc = (model.chan[n, k][model.chan[n, k] >= 0] for k in range(4))
            if len(pc):
                sig[n, pc] = model.dt * sig[n, vc] * masks[n, pc]
            w_act = sig[n, wc][masks[n, wc] > 0]                  # active angular-velocity channels only
            if len(qc) and len(w_act):
                sig[n, qc] = 0.5 * model.dt * w_act.mean() * masks[n, qc]
        return sig

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sigma = None   # history-noise scale [D, S]: the core's one-step error (updated at every check)

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_s=round(time.perf_counter() - start, 1), **extra)
        with (output / "train_log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    runtime = dict(config=cfg, model_config=mc, system=system["name"], training_episodes=int((splits == 0).sum()),
                   validation_episodes=len(val), test_used_for_selection=False, host=platform.node(), **job,
                   torch=torch.__version__, masks=masks.tolist(), stride=stride, dt_s=data["dt"],
                   parameters={n: sum(p.numel() for p in getattr(model, n).parameters()) for n in ("core", "collision", "contact")},
                   data_sha256=file_sha256(args.data / "unified_data.npz"),
                   source_sha256={f.name: file_sha256(f) for f in Path(__file__).parent.glob("*.py")})
    if not (output / "run_config.json").exists():
        atomic_json(output / "run_config.json", runtime)
    else:
        event("resume", **job)
    event("masks", masks=masks.tolist())

    def save(path, stage, update, validation=None):
        torch.save(dict(model_config=mc, system=system, masks=masks.tolist(), model_state_dict=model.state_dict(), runtime=runtime,
                        stage=stage, update=update, validation=validation, sigma=None if sigma is None else sigma.tolist()), path)

    def load_stage(path):
        nonlocal sigma
        packet = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(packet["model_state_dict"])
        sigma = None if packet.get("sigma") is None else torch.tensor(packet["sigma"], dtype=torch.float64, device=dev)
        return packet

    best = {"score": float("inf")}
    checks = {"count": 0}

    def check(stage, update, loss, extra=None):
        was = model.training
        model.eval()
        with torch.no_grad():
            report = free_metrics(model, data, val_check)
        checks["count"] += 1
        if not math.isfinite(report["selection_score"]):
            report["selection_score"] = 1e9
        if report["selection_score"] < best["score"] or not (output / "best.pt").exists():
            best["score"] = report["selection_score"]
            save(output / "best.pt", stage, update, report)
        save(output / "last.pt", stage, update, report)
        event(stage, update=update, loss=float(loss), score=report["selection_score"],
              target_pos_median_mm=1000 * report["target_pos_m"]["median"], target_pos_p95_mm=1000 * report["target_pos_m"]["p95"],
              target_angle_median_deg=math.degrees(report["target_angle_rad"]["median"]),
              target_angle_p95_deg=math.degrees(report["target_angle_rad"]["p95"]),
              path_p95_mm=1000 * report["path_rmse_m"]["p95"], ee_p95_mm=1000 * report["ee_rmse_m"]["p95"],
              moved_pos_median_mm=1000 * report["moved_target_pos_m"]["median"], events_ok=report["event_ok_fraction"],
              finite=report["finite_fraction"], **(extra or {}))
        if was:
            model.train()

    # ---- 1. core -------------------------------------------------------------------
    core = model.core
    nk = len(model.kind_names)
    co = cfg["cotrain"]
    # Free-step windows of the core (every free_phase_step records), grouped per body kind into core_bins bins of
    # log |rate change|; a batch draws from each bin in proportion to the square root of its size.
    phase_keep = (torch.arange(n_win, device=dev) % cfg["free_phase_step"] == 0)[None, :]
    groups, weights = [], []
    triples_by_kind = {}
    for b in range(n_mov):
        w = torch.nonzero(train & phase_keep & ~involved[..., b])
        triples_by_kind.setdefault(int(model.body_kind[b]), []).append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
    for kk, parts in triples_by_kind.items():
        triples = torch.cat(parts)
        b0 = triples[:, 2]
        dch = torch.cat([(states[c[:, 0], c[:, 1] + stride, c[:, 2]] - states[c[:, 0], c[:, 1], c[:, 2]]) for c in triples.split(1 << 20)])
        vel_cols = torch.tensor([[t in ("velocity", "angular_rate") for t in types[b]] + [False] * (smax - len(types[b]))
                                 for b in range(n_mov)], device=dev)[b0]
        dvel = (dch * vel_cols).norm(dim=-1)
        logd = torch.log10(dvel.clamp_min(1e-12))
        sub = logd[torch.randperm(len(logd), device=dev)[:1000000]].float()
        lo, hi = torch.quantile(sub, 0.001).double(), torch.quantile(sub, 0.999).double()
        bins = cfg["core_bins"]
        which = torch.zeros_like(logd, dtype=torch.long) if hi - lo < 1e-6 else ((logd - lo) / (hi - lo) * bins).floor().clamp(0, bins - 1).long()
        for k in range(bins):
            if (which == k).any():
                groups.append(triples[which == k])
                weights.append(math.sqrt(int((which == k).sum())))
    weights = [w / sum(weights) for w in weights]

    def core_batch(n):
        tr = torch.cat([sample(g, max(1, int(round(n * w)))) for g, w in zip(groups, weights)])
        h, ok = hist_of(tr[:, :2])
        h = h[torch.arange(len(tr), device=dev), :, tr[:, 2]][:, :, None]                 # [n, K, 1, S]
        a = actions[tr[:, 0], tr[:, 1], tr[:, 2]][:, None]                               # [n, 1, A]
        target = (states[tr[:, 0], tr[:, 1] + stride, tr[:, 2]] - h[:, -1, 0]) * masks[tr[:, 2]]
        return h, ok, a, tr[:, 2], target

    def core_pred(h, ok, a, body, nz=None):
        """Core of single bodies (each row its own body index)."""
        cur, diff, lag_ok = model.code(h, ok, None)
        if nz is not None:
            diff = diff + torch.randn_like(diff) * nz[body][:, None, None]
        kind = model.body_kind[body]
        present = torch.ones(len(h), 1, dtype=torch.bool, device=dev)
        has = model.has_action[body][:, None]
        fixed = torch.full((len(h),), -1, dtype=torch.long, device=dev)
        out = core(cur, diff, a, present, has, kind[:, None], lag_ok, fixed, kind, model.smask[body][:, None], model.amask[body][:, None])
        out = out.reshape(len(h), nk, smax)[torch.arange(len(h), device=dev), kind]
        return out * masks[body]

    if (output / "core.pt").exists():
        load_stage(output / "core.pt")
        event("core_loaded")
    else:
        with torch.no_grad():
            h, ok, a, body, target = core_batch(128 if smoke else 200000)
            # first-step windows (recorded rates, one history entry) follow another rule (half the rate change);
            # the linear fit of the bulk leaves them out, the network learns them from the one-entry history
            full = ok.sum(-1) >= 2
            if bool(full.any()):
                h, ok, a, body, target = h[full], ok[full], a[full], body[full], target[full]
            for kk in range(nk):
                sel = model.body_kind[body] == kk
                if not sel.any():
                    continue
                cur, diff, act = h[sel, -1, 0], h[sel, 1:, 0] - h[sel, :-1, 0], a[sel, 0]
                bsel = body[sel]
                core.in_mean[kk, 0] = cur.mean(0)
                core.in_std[kk, 0] = torch.maximum(cur.std(0), floor_state[bsel[0]])
                core.diff_std[kk, 0] = torch.maximum(diff[ok[sel][:, :-1]].std(0), floor_diff[bsel[0]])
                core.act_mean[kk, 0] = act.mean(0)
                core.act_std[kk, 0] = torch.maximum(act.std(0), torch.full_like(act[0], 1e-3))
                m_s, m_a = model.smask[bsel[0]], model.amask[bsel[0]]
                zc = (cur - core.in_mean[kk, 0]) / core.in_std[kk, 0] * m_s
                za = (act - core.act_mean[kk, 0]) / core.act_std[kk, 0] * m_a
                design = torch.cat((zc, za, torch.ones_like(zc[:, :1])), -1)
                y = target[sel]
                mean = y.mean(0)
                beta = ridge(design, y - mean, cfg["ridge"])
                resid = y - mean - design @ beta
                rms = y.square().mean(0).sqrt()
                o = slice(kk * smax, (kk + 1) * smax)
                core.out_mean[kk, o] = mean
                core.out_std[kk, o] = torch.maximum(torch.maximum(resid.std(0), 1e-3 * rms), 1e-4 * core.in_std[kk, 0])
                core.lin.data[kk][:, o] = beta / core.out_std[kk, o]
                event("core_linear_fit", kind=model.kind_names[kk], residual_over_rms=(resid.std(0) / rms.clamp_min(1e-30)).tolist())
    # Contact-free validation steps of the core (its one-step error sets the history noise).
    with torch.no_grad():
        pv = []
        for b in range(n_mov):
            w = torch.nonzero(vwin & ~involved[..., b])
            w = w[torch.randperm(len(w), device=dev, generator=rng)[:100000]]
            pv.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
        pv = torch.cat(pv)
        pv_h, pv_ok = hist_of(pv[:, :2])
        pv_h = pv_h[torch.arange(len(pv), device=dev), :, pv[:, 2]][:, :, None]
        pv_a = actions[pv[:, 0], pv[:, 1], pv[:, 2]][:, None]
        pv_t = (states[pv[:, 0], pv[:, 1] + stride, pv[:, 2]] - pv_h[:, -1, 0]) * masks[pv[:, 2]]
        pv_b = pv[:, 2]

    def core_val():
        with torch.no_grad():
            return torch.cat([core_pred(h, o, a, b) for h, o, a, b in
                              zip(pv_h.split(16384), pv_ok.split(16384), pv_a.split(16384), pv_b.split(16384))]) - pv_t

    def core_sigma(err):
        return pose_noise(torch.stack([err[pv_b == b].square().mean(0).sqrt() for b in range(n_mov)]) * masks)

    if not (output / "core.pt").exists():
        err = core_val()   # co-training starts from the linear core; the noise follows that core's error
        sigma = core_sigma(err)
        event("core_sigma", sigma=sigma.tolist(), source="linear core (co-training)")
        with torch.no_grad():
            report = {}
            for b in range(n_mov):
                e = err[pv_b == b].abs()
                report[moving_b[b]["name"]] = {"median": e.median(0).values.tolist(), "p95": torch.quantile(e.float(), 0.95, dim=0).tolist()}
        atomic_json(output / "core_validation.json", report)
        event("core_validation", **{k: v["median"] for k, v in report.items()})
        save(output / "core.pt", "core", 0)
    for p in core.parameters():
        p.requires_grad_(False)
    if sigma is None:
        raise RuntimeError("core noise scale missing")

    # ---- 2. collision -------------------------------------------------------------
    coll = model.collision
    near_rec = max(1, round(cfg["near_negative_s"] / data["record_dt"]))
    near_pair = dilate(labels, near_rec) & train[..., None]

    def pair_raw(windows, pairs_):
        h, ok = hist_of(windows)
        cur, diff, lag_ok = model.code(h, ok)
        rows = torch.arange(len(windows), device=dev)
        return model.pair_inputs(cur, diff, act_of(windows), lag_ok, rows, pairs_)

    def fit_inputs(net, pools):
        """Input standardisation of a pair network, per pair and role position."""
        with torch.no_grad():
            for p, wins in enumerate(pools):
                wins = wins[torch.randperm(len(wins), device=dev, generator=rng)[:50000]]
                c, df, a, present, has, kind, lag_ok, _, _, _, _ = pair_raw(wins, torch.full((len(wins),), p, device=dev))
                for q in range(c.shape[1]):
                    if not present[:, q].any():
                        continue
                    b = int(model.pair_pos[p, q])
                    x = c[present[:, q], q]
                    net.in_mean[p, q] = x.mean(0)
                    net.in_std[p, q] = torch.maximum(x.std(0), floor_state[b])
                    okd = lag_ok[:, 1:] & present[:, q, None]
                    net.diff_std[p, q] = torch.maximum(df[:, :, q][okd].std(0), floor_diff[b])
                    if model.has_action[b]:
                        xa = a[present[:, q], q]
                        net.act_mean[p, q] = xa.mean(0)
                        net.act_std[p, q] = torch.maximum(xa.std(0), torch.full_like(xa[0], 1e-3))

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
        """Per pair: 1/3 contact steps, 1/3 near-contact steps, the rest any step."""
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
        cur, diff, lag_ok = model.code(h, ok, sigma if train_mode else None)
        return model.logits(cur, diff, act_of(picks[:, :2]), lag_ok, torch.arange(len(picks), device=dev), picks[:, 2])

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
                g["lr"] = spec["lr"] * cosine(update, count, spec["warmup"], 0.01)
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

    # ---- 3. contact network, co-trained with the core -----------------------------
    cont = model.contact
    hit_windows = torch.nonzero(train & labels.any(-1))   # contact batches; rollout starts lie 1..horizon steps before

    def own_history(windows, m):
        """Histories whose last m entries are the model's own steps (recorded switches and commands)."""
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
                tt = t0[idx] + step * stride
                g = labels[windows[idx, 0], tt].to(torch.float64)
                nxt = model.step_with_gates(h[idx], ok[idx], actions[windows[idx, 0], tt], g)
                h2, ok2 = model.advance(h[idx], ok[idx], nxt)
                h, ok = h.clone(), ok.clone()
                h[idx], ok[idx] = h2, ok2
        return h, ok

    push_stats = {"tried": 0, "kept": 0}

    def contact_example(windows, train_mode, push=0.0, core_grad=False):
        """One step with the recorded switches; with probability `push` the history is the model's own (kept only
        if its last state lies within push_max_dev one-step RMS changes of the record)."""
        h, ok = hist_of(windows)
        if push > 0:
            sel = torch.nonzero(torch.rand(len(windows), device=dev, generator=rng) < push).flatten()
            if len(sel):
                m = torch.randint(1, cfg["push_max_steps"] + 1, (len(sel),), device=dev, generator=rng)
                hs, oks = own_history(windows[sel], m)
                dev_ = ((hs[:, -1] - h[sel, -1]) * masks / step_rms).abs().amax((-1, -2))
                keep = dev_ <= cfg["push_max_dev"]
                push_stats["tried"] += len(sel)
                push_stats["kept"] += int(keep.sum())
                sel, hs, oks = sel[keep], hs[keep], oks[keep]
                h, ok = h.clone(), ok.clone()
                h[sel], ok[sel] = hs, oks
        if train_mode:
            ok = drop_history(ok)
        g = labels[windows[:, 0], windows[:, 1]].to(torch.float64)
        pred = model.step_with_gates(h, ok, act_of(windows), g, sigma if train_mode else None, core_grad=core_grad)
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
            # Output scale and linear path per pair from steps with exactly one contact (attributed to that pair).
            gt = labels[hit_windows[:, 0], hit_windows[:, 1]]
            single = hit_windows[gt.sum(-1) == 1]
            single = single[torch.randperm(len(single), device=dev, generator=rng)[:400000]]
            sp = labels[single[:, 0], single[:, 1]].float().argmax(-1)
            h, ok = hist_of(single)
            act = act_of(single)
            cur, diff, lag_ok = model.code(h, ok)
            core_d = torch.cat([model.run_core(*t) for t in zip(cur.split(65536), diff.split(65536), act.split(65536), lag_ok.split(65536))])
            resid = (states[single[:, 0], single[:, 1] + stride] - cur - core_d) * masks           # [n, D, S]
            fit_resid = resid.clone()   # integrated pose: the contact network's pose output is not used
            for n in torch.nonzero(model.integ_body).flatten().tolist():
                pc_ = model.chan[n, 0][model.chan[n, 0] >= 0]
                if len(pc_):
                    fit_resid[:, n, pc_] = 0.0
            c, df, a, present, has, kind, lo, fx, q, sm, am = model.pair_inputs(cur, diff, act, lag_ok, torch.arange(len(single), device=dev), sp)
            zc, _, za = cont.scaled(c, df, a, present, has, lo, q, sm, am)
            design = torch.cat((zc.flatten(1), za.flatten(1), torch.ones_like(zc[:, :1, 0])), -1)
            rr = torch.arange(len(sp), device=dev)
            part2 = torch.where(model.pair_mm[sp][:, None], fit_resid[rr, model.pair_pos[sp, 1].clamp_min(0)], torch.zeros_like(fit_resid[:, 0]))
            tgt = torch.cat((fit_resid[rr, model.pair_pos[sp, 0]], part2), -1)                                # [n, 2S]
            fit_report, present_pairs = {}, []
            for p in range(n_pairs):
                sel = sp == p
                if sel.sum() < 20:
                    continue
                present_pairs.append(p)
                x, y = design[sel], tgt[sel]
                mean = y.mean(0)
                beta = ridge(x, y - mean, cfg["skip_ridge"])
                r = y - mean - x @ beta
                rms = y.square().mean(0).sqrt()
                scale_in = torch.cat((cont.in_std[p, 0], cont.in_std[p, 1] if model.pair_mm[p] else cont.in_std[p, 0]))
                cont.out_mean[p] = mean
                cont.out_std[p] = torch.maximum(torch.maximum(r.std(0), 1e-3 * rms), 1e-4 * scale_in)
                cont.lin.data[p] = beta / cont.out_std[p]
                fit_report[f"pair{p}"] = dict(steps=int(sel.sum()), linear_r2=float(1 - r.square().sum() / (y - mean).square().sum().clamp_min(1e-30)))
            for p in range(n_pairs):
                if p not in present_pairs:
                    cont.out_std[p] = cont.out_std[present_pairs].mean(0)
            # loss scale per body and channel: kappa(type) x RMS of that body's contact change
            sc = torch.ones(n_mov, smax, dtype=torch.float64, device=dev)
            for b in range(n_mov):
                rows = []
                for p in present_pairs:
                    sel = sp == p
                    if int(model.pair_pos[p, 0]) == b:
                        rows.append(resid[sel][:, b])
                    if model.pair_mm[p] and int(model.pair_pos[p, 1]) == b:
                        rows.append(resid[sel][:, b])
                if rows:
                    rms = torch.cat(rows).square().mean(0).sqrt()
                    sc[b] = torch.where(masks[b] > 0, torch.maximum(kappa[b] * rms, 1e-3 * kappa[b] * floor_state[b]), torch.ones_like(rms))
            # RMS of the full one-step change (y - x) of each affected body's channels on contact steps: the unit
            # of the own-history guard (a deviation is judged against how much the channel moves in one step).
            step = (states[single[:, 0], single[:, 1] + stride] - cur) * masks
            srms = torch.ones(n_mov, smax, dtype=torch.float64, device=dev)
            for b in range(n_mov):
                touched_b = (model.pair_pos[sp, 0] == b) | (model.pair_mm[sp] & (model.pair_pos[sp, 1] == b))
                if touched_b.any():
                    r = step[touched_b, b].square().mean(0).sqrt()
                    srms[b] = torch.where(masks[b] > 0, r.clamp_min(1e-12), torch.ones_like(r))
        event("contact_linear_fit", loss_scale=sc.tolist(), step_rms=srms.tolist(), **fit_report)
        (output / "contact_scale.json").write_text(json.dumps({"scale": sc.tolist(), "step_rms": srms.tolist()}))

    _sc = json.loads((output / "contact_scale.json").read_text())
    scale = torch.tensor(_sc["scale"], dtype=torch.float64, device=dev)
    step_rms = torch.tensor(_sc["step_rms"], dtype=torch.float64, device=dev)
    vpos = torch.nonzero(vwin & labels.any(-1))
    vpos = vpos[torch.randperm(len(vpos), device=dev, generator=rng)[: (200 if smoke else 20000)]]

    def onestep():
        """One-step contact error on validation contact steps (recorded and own histories), in loss scales."""
        with torch.no_grad():
            out = {}
            for name, push in (("true", 0.0), ("own", 1.0)):
                errs = []
                for c in vpos.split(4096):
                    pred, y, aff, _ = contact_example(c, False, push)
                    e = ((pred - y) * core_masks / scale).abs().amax(-1)                              # [n, D]
                    errs.append(e[aff > 0])
                e = torch.cat(errs) if errs else torch.zeros(1, dtype=torch.float64, device=dev)
                out[f"onestep_{name}_scaled_median"] = float(e.median())
                out[f"onestep_{name}_scaled_p95"] = float(torch.quantile(e.float(), 0.95))
            return out

    def contact_loss(windows, push, train_mode=True, core_grad=False):
        pred, y, aff, _ = contact_example(windows, train_mode, push, core_grad)
        err = ((pred - y) * core_masks / scale).square().sum(-1) / core_masks.sum(-1).clamp_min(1)
        return (err * aff).sum() / aff.sum().clamp_min(1)

    rscale = scale * cfg["rollout_scale_factor"]

    def rollout_loss(horizon, batch):
        """Huber loss of free rollouts (the model's own states, recorded commands) against the recorded trajectory;
        half of the starts lie 1..horizon steps before a contact step."""
        starts = sample(hit_windows, batch // 2).clone()
        starts[:, 1] = (starts[:, 1] - torch.randint(1, max(2, horizon), (len(starts),), device=dev, generator=rng) * stride).clamp_min(0)
        w = torch.cat((starts, sample(any_train, batch - len(starts))))
        h, ok = hist_of(w)
        current = h[:, -1]
        total, den = current.sum() * 0.0, 0
        for step in range(horizon):
            t_now = w[:, 1] + step * stride
            j = t_now + stride
            active = j < lengths[w[:, 0]]
            y = states[w[:, 0], j.clamp_max(n_rec - 1)]
            pred = model(h, ok, actions[w[:, 0], t_now.clamp_max(n_rec - 1)])
            err = F.huber_loss((pred - y) * masks / rscale, torch.zeros_like(pred), reduction="none", delta=1.0)
            total = total + (err.mean((-1, -2)) * active).sum()
            den += int(active.sum())
            current = torch.where(active[:, None, None], pred, current)
            h, ok = model.advance(h, ok, current)
        return total / max(1, den)

    if not (output / "contact_done.json").exists():
        # L_core + L_contact + L_trajectory, each divided by its running mean
        for p in core.parameters():
            p.requires_grad_(True)
        joint = [p for p in list(core.parameters()) + list(cont.parameters()) if p.requires_grad]
        count = 5 if smoke else co["updates"]
        opt = torch.optim.AdamW(joint, lr=co["lr"], weight_decay=0.0)
        first, ema = 1, {}
        if resume_path.exists():
            state = torch.load(resume_path, map_location=dev, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            opt.load_state_dict(state["optimizer"])
            rng.set_state(state["rng"].cpu())
            best["score"], first, ema = state["best"], state["update"] + 1, state["ema"]
            sigma = torch.tensor(state["sigma"], dtype=torch.float64, device=dev)
            event("cotrain_resumed", update=state["update"], best=best["score"])
        h0 = max(1, round(co["horizon_start_s"] / model.dt))
        h1 = max(h0, round(co["horizon_max_s"] / model.dt))
        push_from = int(cfg["push_start"] * count)
        traj_every = co["traj_every"]
        for update in range(first, count + 1):
            frac = update / count
            for g in opt.param_groups:
                g["lr"] = co["lr"] * cosine(update, count, co["warmup"], 0.01)
            hb, okb, ab, bb, tb = core_batch(64 if smoke else co["core_batch"])
            okb = drop_history(okb)
            kind = model.body_kind[bb]
            kscale = torch.stack([core.out_std[k, k * smax:(k + 1) * smax] for k in range(nk)])[kind]
            terms = {"core": (((core_pred(hb, okb, ab, bb, sigma) - tb) / kscale).square() * core_masks[bb]).sum() / core_masks[bb].sum()}
            push = cfg["push_prob"] if update > push_from else 0.0
            terms["contact"] = contact_loss(sample(hit_windows, 64 if smoke else co["contact_batch"]), push, core_grad=True)
            if update % traj_every == 0:
                horizon = 4 if smoke else round(h0 + (h1 - h0) * min(1.0, frac / co["horizon_ramp"]))
                terms["traj"] = rollout_loss(horizon, 4 if smoke else co["traj_batch"])
            total = 0.0
            for name, value in terms.items():
                v = float(value.detach())
                ema[name] = v if name not in ema else 0.99 * ema[name] + 0.01 * v
                weight = traj_every * min(1.0, frac / co["traj_ramp"]) if name == "traj" else 1.0
                total = total + weight * value / max(ema[name], 1e-30)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(joint, 1.0)
            if not torch.isfinite(gnorm):
                event("cotrain_nonfinite_gradient", update=update)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            if update % (5 if smoke else cfg["evaluate_every"]) == 0 or update == count:
                sigma = core_sigma(core_val())   # the noise follows the current core's one-step error
                check("cotrain", update, total.detach(), {**onestep(), **{f"ema_{k}": v for k, v in ema.items()},
                                                          "push_kept_fraction": push_stats["kept"] / max(1, push_stats["tried"])})
                push_stats.update(tried=0, kept=0)
                tmp = output / "contact_state.pt.tmp"
                torch.save(dict(model_state_dict=model.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                                best=best["score"], update=update, ema=ema, sigma=sigma.tolist()), tmp)
                tmp.replace(resume_path)
        (output / "contact_done.json").write_text(json.dumps({"best": best["score"]}))

    # ---- 4. final validation of best.pt -------------------------------------------
    if not (output / "complete.json").exists():
        model_best, metadata = load_model(output / "best.pt", dev)
        with torch.no_grad():
            report = free_metrics(model_best, data, val)
        atomic_json(output / "validation.json", report)
        done = dict(complete=True, smoke=smoke, test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                    checkpoint_sha256=file_sha256(output / "best.pt"), best_stage=metadata["stage"],
                    best_update=metadata["update"], validation_score=report["selection_score"], checkpoints_compared=checks["count"],
                    target_pos_median_mm=1000 * report["target_pos_m"]["median"], target_pos_p95_mm=1000 * report["target_pos_m"]["p95"],
                    target_angle_median_deg=math.degrees(report["target_angle_rad"]["median"]),
                    target_angle_p95_deg=math.degrees(report["target_angle_rad"]["p95"]),
                    moved_pos_median_mm=1000 * report["moved_target_pos_m"]["median"],
                    moved_angle_median_deg=math.degrees(report["moved_target_angle_rad"]["median"]),
                    path_p95_mm=1000 * report["path_rmse_m"]["p95"], ee_p95_mm=1000 * report["ee_rmse_m"]["p95"],
                    events_ok=report["event_ok_fraction"], finite=report["finite_fraction"])
        atomic_json(output / "complete.json", done)
        event("complete", **done)


if __name__ == "__main__":
    main()
