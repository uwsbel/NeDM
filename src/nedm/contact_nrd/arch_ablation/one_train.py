"""The training of a one-network model (1T or 1M): the 3T's co-training with one network in place of three.

Called by ``python -m nedm.contact_nrd.arch_ablation.train --arch 1t|1m``. Sampling, the three loss terms (core,
contact, rollout), history noise and dropout, pushes, the optimiser, the schedule and the checks are those of
``nedm.contact_nrd.train``; the code below is that file's, with these differences:

* there is no collision stage, and the recorded contact switches are not an input (they still choose the contact
  windows);
* the network reads the K states *and the K commands* of the history (the 3T reads only the newest command), one
  token per step holding every moving body;
* its output is the change of every channel. As in the 3T, integrated poses are not taken from it:
  ``assemble`` integrates them from the new rates;
* the core term is scaled by this network's own target standard deviation (the 3T: its core's ``out_std``), and
  the history noise starts from this network's own error;
* it predicts no contacts, so its checks use the **motion score**: the selection score without its contact-event
  terms (common.motion_score).

A 3T is built and its setup fits run once (the core's linear fit and the contact scale), because the windows, the
loss scale and the push guard come from them; it is never trained. ``--updates`` sets the number of co-training
updates (default: the config's; 150000 gives the 3T's collision + co-training total). By default the network has
as many parameters as the 3T; ``--width`` and ``--parameters`` change that.

Output directory: run_config.json, train_log.jsonl, setup.pt, state.pt (resume state), best.pt, last.pt,
contact_done.json, validation.json, complete.json.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from nedm.contact_nrd import train as base
from nedm.contact_nrd.arch_ablation import checkpoints, common, one_net
from nedm.contact_nrd.data import (atomic_json, dilate, evaluate, file_sha256, load_data, selected_episodes,
                                   step_average_rates, window_labels)
from nedm.contact_nrd.model import ContactNRD, model_config
from nedm.contact_nrd.train import TYPE_RULES, cosine, ridge


def train_one(args):
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = {**json.loads(args.config.read_text()), "arch": args.arch}
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.updates is not None:
        cfg["cotrain"] = {**cfg["cotrain"], "updates": args.updates}
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
    family = data["family"]
    if smoke:   # 24 episodes; with families up to 8 of each
        val = val[:24] if family is None else torch.cat([val[family[val] == k][:8] for k in range(len(system["families"]))])
        val = torch.sort(val).values
    # Checks during training use an evenly spaced subset of the validation episodes (of the families with
    # select: true); the final report uses all.
    pick = selected_episodes(data, val)
    if not len(pick):
        raise ValueError("no validation episodes in the families that checkpoint selection uses")
    n_check = min(len(pick), int(cfg["check_episodes"]))
    val_check = pick[torch.linspace(0, len(pick) - 1, n_check, device=dev).round().long()]
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
    explicit = [d for d, b in enumerate(moving_b) if "channel_mask" in b]
    for d in explicit:   # the data's own mask (e.g. a robot's base height, small next to its travel)
        cm = [float(x) for x in moving_b[d]["channel_mask"]]
        if len(cm) != len(types[d]) or set(cm) - {0.0, 1.0}:
            raise ValueError(f"{moving_b[d]['name']}: channel_mask needs one 0/1 per channel")
        masks[d] = 0.0
        masks[d, : len(cm)] = torch.tensor(cm, dtype=torch.float64, device=dev)

    mc = model_config(data["dt"], K, cfg["network"], cfg["integrate_kinds"])
    three = ContactNRD(mc, system, masks.tolist()).to(dev)   # the setup fits and the step's bookkeeping; never trained
    for p in three.parameters():
        p.requires_grad_(False)
    # The integrator pairs pose and rate channels by order, so their masks must agree; a quaternion's mask is
    # [any angular axis, x, y, z] of its angular velocity mask.
    for n in torch.nonzero(three.integ_body).flatten().tolist():
        pc, qc, vc, wc = (three.chan[n, k][three.chan[n, k] >= 0] for k in range(4))
        m = masks[n]
        agree = torch.equal(m[pc], m[vc])
        if len(qc) == 4 and len(wc) == 3:
            agree = agree and torch.equal(m[qc], torch.cat((m[wc].amax()[None], m[wc])))
        if not agree and n in explicit:
            raise ValueError(f"{moving_b[n]['name']}: channel_mask must give a pose channel and its rate channel "
                             "the same value (quaternion: [any angular axis, x, y, z])")
    # Channels that are trained (core and contact losses): integrated pose channels are computed, not predicted.
    core_masks = masks.clone()
    for b in torch.nonzero(three.integ_body).flatten().tolist():
        for k in (0, 1):
            cols = three.chan[b, k][three.chan[b, k] >= 0]
            core_masks[b, cols] = 0.0

    def pose_noise(sig):
        """History noise of integrated pose channels from the noise of their rates: a rate error s changes the next
        pose step by dt s (quaternion: 0.5 dt s); the pose outputs are not trained and do not give this noise."""
        sig = sig.clone()
        for n in torch.nonzero(three.integ_body).flatten().tolist():
            pc, qc, vc, wc = (three.chan[n, k][three.chan[n, k] >= 0] for k in range(4))
            if len(pc):
                sig[n, pc] = three.dt * sig[n, vc] * masks[n, pc]
            w_act = sig[n, wc][masks[n, wc] > 0]                  # active angular-velocity channels only
            if len(qc) and len(w_act):
                sig[n, qc] = 0.5 * three.dt * w_act.mean() * masks[n, qc]
        return sig

    # ---- the one network ---------------------------------------------------------------
    any_train = torch.nonzero(train)
    state_dim = n_mov * smax   # every moving body of a step is one token
    arch = checkpoints.ARCHS[args.arch]   # arch_1t or arch_1m: the sizes and the network
    sizes = arch.config(K, state_dim, three.amax, target=args.parameters or common.count(three), width=args.width)
    one = arch.build(sizes, K, state_dim, three.amax).to(dev)
    with torch.no_grad():   # statistics of up to 200 000 training windows
        g = torch.Generator(device=dev).manual_seed(cfg["seed"])
        w = any_train[torch.randperm(len(any_train), device=dev, generator=g)[:200000]]
        s = states[w[:, 0], w[:, 1]].flatten(1)
        one.set_statistics(s.float(), actions[w[:, 0], w[:, 1], 0].float(),
                           (states[w[:, 0], w[:, 1] + stride].flatten(1) - s).float(), three.smask.flatten().float())
    net = one_net.OneNetNRD(one, three)
    one_evaluate = one_net.motion_evaluate(evaluate, net)   # the checks: motion score, earlier commands supplied

    def hist_of(windows):
        net.set_past(data, windows[:, 0], windows[:, 1])
        raw = windows[:, 1, None] + torch.arange(1 - K, 1, device=dev) * stride
        ok = raw >= 0
        h = states[windows[:, 0, None], raw.clamp_min(0)] * ok[..., None, None]
        return h, ok

    def act_of(windows):
        net.set_past(data, windows[:, 0], windows[:, 1])
        return actions[windows[:, 0], windows[:, 1]]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sigma = None   # history-noise scale [D, S]: the network's one-step error (updated at every check)

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_s=round(time.perf_counter() - start, 1), **extra)
        with (output / "train_log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    contact_nrd = Path(base.__file__).parent
    runtime = dict(config=cfg, model_config=mc, system=system["name"], training_episodes=int((splits == 0).sum()),
                   validation_episodes=len(val), test_used_for_selection=False, host=platform.node(), **job,
                   torch=torch.__version__, masks=masks.tolist(), stride=stride, dt_s=data["dt"],
                   one_config=sizes, parameters=common.count(one), reference_3t_parameters=common.count(three),
                   selection="motion score", data_sha256=file_sha256(args.data / "unified_data.npz"),
                   source_sha256={str(f.relative_to(contact_nrd)): file_sha256(f)   # arch_ablation/*.py and *.py
                                  for d in (Path(__file__).parent, contact_nrd) for f in sorted(d.glob("*.py"))})
    if not (output / "run_config.json").exists():
        atomic_json(output / "run_config.json", runtime)
    else:
        event("resume", **job)
    event("masks", masks=masks.tolist(), one_config=sizes, parameters=runtime["parameters"])

    def save(path, stage, update, validation=None):
        torch.save(dict(arch=args.arch, model_config=mc, system=system, masks=masks.tolist(), one_config=sizes,
                        model_state_dict=one.state_dict(), runtime=runtime, stage=stage, update=update,
                        validation=validation, sigma=None if sigma is None else sigma.tolist()), path)

    best = {"score": float("inf")}
    checks = {"count": 0}

    def check(stage, update, loss, extra=None):
        was = net.training
        net.eval()
        with torch.no_grad():
            report = one_evaluate(net, data, val_check, score_only=True)
        checks["count"] += 1
        if not math.isfinite(report["selection_score"]):
            report["selection_score"] = 1e9
        if report["selection_score"] < best["score"] or not (output / "best.pt").exists():
            best["score"] = report["selection_score"]
            save(output / "best.pt", stage, update, report)
        save(output / "last.pt", stage, update, report)
        if "evaluation" in system:
            event(stage, update=update, loss=float(loss), score=report["selection_score"],
                  terms={t["stat"]: t["value"] for t in report["score_terms"]}, finite=report["finite_fraction"],
                  **(extra or {}))
        else:
            event(stage, update=update, loss=float(loss), score=report["selection_score"],
                  target_pos_median_mm=1000 * report["target_pos_m"]["median"], target_pos_p95_mm=1000 * report["target_pos_m"]["p95"],
                  target_angle_median_deg=math.degrees(report["target_angle_rad"]["median"]),
                  target_angle_p95_deg=math.degrees(report["target_angle_rad"]["p95"]),
                  path_p95_mm=1000 * report["path_rmse_m"]["p95"], ee_p95_mm=1000 * report["ee_rmse_m"]["p95"],
                  moved_pos_median_mm=1000 * report["moved_target_pos_m"]["median"], finite=report["finite_fraction"],
                  **(extra or {}))
        if was:
            net.train()

    # ---- 1. the 3T's setup fits: core --------------------------------------------------
    core = three.core
    nk = len(three.kind_names)
    co = cfg["cotrain"]
    # Free-step windows of the core (every free_phase_step records), grouped per body kind into core_bins bins of
    # log |rate change|; a batch draws from each bin in proportion to the square root of its size.
    phase_keep = (torch.arange(n_win, device=dev) % cfg["free_phase_step"] == 0)[None, :]
    groups, weights = [], []
    triples_by_kind = {}
    for b in range(n_mov):
        w = torch.nonzero(train & phase_keep & ~involved[..., b])
        triples_by_kind.setdefault(int(three.body_kind[b]), []).append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
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

    def core_rows(n):
        """(episode, record, body) rows of contact-free steps: the 3T's core batch (the same draws)."""
        return torch.cat([sample(g, max(1, int(round(n * w)))) for g, w in zip(groups, weights)])

    def body_window(rows):
        """Rows -> every body's history, ok, every body's commands, the body, and its masked one-step change: the
        one network reads all bodies, and its row of the sampled body is scored."""
        h, ok = hist_of(rows[:, :2])
        e, t, body = rows.unbind(1)
        target = (states[e, t + stride, body] - states[e, t, body]) * masks[body]
        return h, ok, actions[e, t], body, target

    def core_pred(h, ok, a, body, nz=None):
        change = net.change(h, ok, a, nz)
        return change[torch.arange(len(h), device=dev), body] * masks[body]

    setup_path = output / "setup.pt"
    if not setup_path.exists():
        with torch.no_grad():   # the core's linear path: a ridge fit per body kind on contact-free steps
            tr = core_rows(128 if smoke else 200000)
            raw = tr[:, 1, None] + torch.arange(1 - K, 1, device=dev) * stride
            ok = raw >= 0
            h = (states[tr[:, 0, None], raw.clamp_min(0)] * ok[..., None, None])[torch.arange(len(tr), device=dev), :, tr[:, 2]][:, :, None]
            a = actions[tr[:, 0], tr[:, 1], tr[:, 2]][:, None]
            body = tr[:, 2]
            target = (states[tr[:, 0], tr[:, 1] + stride, body] - h[:, -1, 0]) * masks[body]
            # first-step windows (recorded rates, one history entry) follow another rule (half the rate change);
            # the linear fit of the bulk leaves them out
            full = ok.sum(-1) >= 2
            if bool(full.any()):
                h, ok, a, body, target = h[full], ok[full], a[full], body[full], target[full]
            for kk in range(nk):
                sel = three.body_kind[body] == kk
                if not sel.any():
                    continue
                cur, diff, act = h[sel, -1, 0], h[sel, 1:, 0] - h[sel, :-1, 0], a[sel, 0]
                bsel = body[sel]
                core.in_mean[kk, 0] = cur.mean(0)
                core.in_std[kk, 0] = torch.maximum(cur.std(0), floor_state[bsel[0]])
                core.diff_std[kk, 0] = torch.maximum(diff[ok[sel][:, :-1]].std(0), floor_diff[bsel[0]])
                core.act_mean[kk, 0] = act.mean(0)
                core.act_std[kk, 0] = torch.maximum(act.std(0), torch.full_like(act[0], 1e-3))
                m_s, m_a = three.smask[bsel[0]], three.amask[bsel[0]]
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
                event("core_linear_fit", kind=three.kind_names[kk], residual_over_rms=(resid.std(0) / rms.clamp_min(1e-30)).tolist())
    # Contact-free validation steps (the network's one-step error on them sets the history noise).
    with torch.no_grad():
        pv = []
        for b in range(n_mov):
            w = torch.nonzero(vwin & ~involved[..., b])
            w = w[torch.randperm(len(w), device=dev, generator=rng)[:100000]]
            pv.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
        pv = torch.cat(pv)
        pv_b = pv[:, 2]

    def core_val():
        errors = []
        with torch.no_grad():
            for c in pv.split(16384):   # each batch with its own earlier commands
                h, ok, a, body, target = body_window(c)
                errors.append(core_pred(h, ok, a, body) - target)
        return torch.cat(errors)

    # Optional history-noise rule (config "history_noise", absent in train.json = the released rule): the noise is
    # multiplied by "scale" and capped per channel at "cap_step_rms" x the RMS of the one-step change on training
    # steps with one active pair.
    noise_rule = cfg.get("history_noise") or {}
    noise_scale, noise_cap = float(noise_rule.get("scale", 1.0)), noise_rule.get("cap_step_rms")
    if noise_cap is not None:
        with torch.no_grad():
            first = torch.nonzero(train & (labels.sum(-1) == 1))
            first = first[torch.linspace(0, len(first) - 1, min(len(first), 400000), device=dev).round().long()]
            on = labels[first[:, 0], first[:, 1]]
            change = (states[first[:, 0], first[:, 1] + stride] - states[first[:, 0], first[:, 1]]) * masks
            cap_rms = torch.full((n_mov, smax), float("inf"), dtype=torch.float64, device=dev)
            for b in range(n_mov):
                hit = torch.zeros(len(first), dtype=torch.bool, device=dev)
                for p, (i, j) in enumerate(pairs):
                    if slot[i] == b or slot.get(j, -1) == b:
                        hit |= on[:, p]
                if hit.any():
                    cap_rms[b] = float(noise_cap) * change[hit, b].square().mean(0).sqrt()

    def core_sigma(err):
        rms = torch.stack([err[pv_b == b].square().mean(0).sqrt() for b in range(n_mov)])
        if noise_cap is not None:
            rms = torch.minimum(rms, cap_rms)
        return pose_noise(rms * masks) * noise_scale if noise_rule else pose_noise(rms * masks)

    # ---- 2. the 3T's setup fits: pair inputs, contact scale ----------------------------
    near_rec = max(1, round(cfg["near_negative_s"] / data["record_dt"]))
    near_pair = dilate(labels, near_rec) & train[..., None]

    def pair_raw(windows, pairs_):
        h, ok = hist_of(windows)
        cur, diff, lag_ok = three.code(h, ok)
        rows = torch.arange(len(windows), device=dev)
        return three.pair_inputs(cur, diff, act_of(windows), lag_ok, rows, pairs_)

    def fit_inputs(tnet, pools):
        """Input standardisation of a pair network, per pair and role position."""
        with torch.no_grad():
            for p, wins in enumerate(pools):
                wins = wins[torch.randperm(len(wins), device=dev, generator=rng)[:50000]]
                c, df, a, present, has, kind, lag_ok, _, _, _, _ = pair_raw(wins, torch.full((len(wins),), p, device=dev))
                for q in range(c.shape[1]):
                    if not present[:, q].any():
                        continue
                    b = int(three.pair_pos[p, q])
                    x = c[present[:, q], q]
                    tnet.in_mean[p, q] = x.mean(0)
                    tnet.in_std[p, q] = torch.maximum(x.std(0), floor_state[b])
                    okd = lag_ok[:, 1:] & present[:, q, None]
                    tnet.diff_std[p, q] = torch.maximum(df[:, :, q][okd].std(0), floor_diff[b])
                    if three.has_action[b]:
                        xa = a[present[:, q], q]
                        tnet.act_mean[p, q] = xa.mean(0)
                        tnet.act_std[p, q] = torch.maximum(xa.std(0), torch.full_like(xa[0], 1e-3))

    hit_windows = torch.nonzero(train & labels.any(-1))   # contact batches; rollout starts lie 1..horizon steps before
    if setup_path.exists():
        packet = torch.load(setup_path, map_location=dev, weights_only=False)
        three.load_state_dict(packet["three"])
        sc, srms = packet["scale"], packet["step_rms"]
        event("setup_loaded")
    else:
        cont = three.contact
        with torch.no_grad():
            pools = []
            for p in range(n_pairs):
                w = torch.nonzero(near_pair[..., p])
                pools.append(w if len(w) > 1000 else torch.cat((w, sample(any_train, 50000))))
            fit_inputs(three.collision, pools)   # not used by a prediction; kept for the 3T's order of random draws
            cpools = []
            for p in range(n_pairs):
                w = torch.nonzero(labels[..., p] & train)
                cpools.append(w if len(w) > 200 else pools[p])
            fit_inputs(cont, cpools)
            # Steps with exactly one contact (attributed to that pair): the contact change after the linear core.
            gt = labels[hit_windows[:, 0], hit_windows[:, 1]]
            single = hit_windows[gt.sum(-1) == 1]
            single = single[torch.randperm(len(single), device=dev, generator=rng)[:400000]]
            sp = labels[single[:, 0], single[:, 1]].float().argmax(-1)
            h, ok = hist_of(single)
            act = act_of(single)
            cur, diff, lag_ok = three.code(h, ok)
            core_d = torch.cat([three.run_core(*t) for t in zip(cur.split(65536), diff.split(65536), act.split(65536), lag_ok.split(65536))])
            resid = (states[single[:, 0], single[:, 1] + stride] - cur - core_d) * masks           # [n, D, S]
            # (the 3T also fits its contact network's linear path here; a one-network model has none)
            present_pairs = [p for p in range(n_pairs) if (sp == p).sum() >= 20]
            # loss scale per body and channel: kappa(type) x RMS of that body's contact change
            sc = torch.ones(n_mov, smax, dtype=torch.float64, device=dev)
            for b in range(n_mov):
                rows = []
                for p in present_pairs:
                    sel = sp == p
                    if int(three.pair_pos[p, 0]) == b:
                        rows.append(resid[sel][:, b])
                    if three.pair_mm[p] and int(three.pair_pos[p, 1]) == b:
                        rows.append(resid[sel][:, b])
                if rows:
                    rms = torch.cat(rows).square().mean(0).sqrt()
                    sc[b] = torch.where(masks[b] > 0, torch.maximum(kappa[b] * rms, 1e-3 * kappa[b] * floor_state[b]), torch.ones_like(rms))
            # RMS of the full one-step change (y - x) of each affected body's channels on contact steps: the unit
            # of the own-history guard (a deviation is judged against how much the channel moves in one step).
            step = (states[single[:, 0], single[:, 1] + stride] - cur) * masks
            srms = torch.ones(n_mov, smax, dtype=torch.float64, device=dev)
            for b in range(n_mov):
                touched_b = (three.pair_pos[sp, 0] == b) | (three.pair_mm[sp] & (three.pair_pos[sp, 1] == b))
                if touched_b.any():
                    r = step[touched_b, b].square().mean(0).sqrt()
                    srms[b] = torch.where(masks[b] > 0, r.clamp_min(1e-12), torch.ones_like(r))
        sc, srms = sc.tolist(), srms.tolist()
        event("contact_scale", loss_scale=sc, step_rms=srms)
        torch.save(dict(three=three.state_dict(), scale=sc, step_rms=srms), setup_path)
    scale = torch.tensor(sc, dtype=torch.float64, device=dev)
    step_rms = torch.tensor(srms, dtype=torch.float64, device=dev)

    # ---- 3. co-training of the one network ---------------------------------------------
    def own_history(windows, m):
        """Histories whose last m entries are the model's own steps (the recorded commands before each)."""
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
                net.set_past(data, windows[idx, 0], tt)
                nxt = net.next_state(h[idx], ok[idx], actions[windows[idx, 0], tt])
                h2, ok2 = net.advance(h[idx], ok[idx], nxt)
                h, ok = h.clone(), ok.clone()
                h[idx], ok[idx] = h2, ok2
        return h, ok

    push_stats = {"tried": 0, "kept": 0}

    def contact_example(windows, train_mode, push=0.0):
        """One step on a contact window; with probability `push` the history is the model's own (kept only if its
        last state lies within push_max_dev one-step RMS changes of the record)."""
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
        g = labels[windows[:, 0], windows[:, 1]]
        pred = net.next_state(h, ok, act_of(windows), sigma if train_mode else None)
        y = states[windows[:, 0], windows[:, 1] + stride]
        affected = torch.zeros(len(windows), n_mov, dtype=torch.float64, device=dev)
        for p in range(n_pairs):
            on = g[:, p]
            affected[on, three.pair_pos[p, 0]] = 1.0
            if three.pair_mm[p]:
                affected[on, three.pair_pos[p, 1]] = 1.0
        return pred, y, affected

    vpos = torch.nonzero(vwin & labels.any(-1))
    vpos = vpos[torch.randperm(len(vpos), device=dev, generator=rng)[: (200 if smoke else 20000)]]

    def onestep():
        """One-step contact error on validation contact steps (recorded and own histories), in loss scales."""
        with torch.no_grad():
            out = {}
            for name, push in (("true", 0.0), ("own", 1.0)):
                errs = []
                for c in vpos.split(4096):
                    pred, y, aff = contact_example(c, False, push)
                    e = ((pred - y) * core_masks / scale).abs().amax(-1)                              # [n, D]
                    errs.append(e[aff > 0])
                e = torch.cat(errs) if errs else torch.zeros(1, dtype=torch.float64, device=dev)
                out[f"onestep_{name}_scaled_median"] = float(e.median())
                out[f"onestep_{name}_scaled_p95"] = float(torch.quantile(e.float(), 0.95))
            return out

    def contact_loss(windows, push):
        pred, y, aff = contact_example(windows, True, push)
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
            pred = net(h, ok, actions[w[:, 0], t_now.clamp_max(n_rec - 1)])
            err = F.huber_loss((pred - y) * masks / rscale, torch.zeros_like(pred), reduction="none", delta=1.0)
            total = total + (err.mean((-1, -2)) * active).sum()
            den += int(active.sum())
            current = torch.where(active[:, None, None], pred, current)
            h, ok = net.advance(h, ok, current)
        return total / max(1, den)

    resume_path = output / "state.pt"
    if (output / "contact_done.json").exists():
        best["score"] = json.loads((output / "contact_done.json").read_text())["best"]
        event("cotrain_loaded", best=best["score"])
    else:
        # L_core + L_contact + L_trajectory, each divided by its running mean
        params = list(one.parameters())
        count = 5 if smoke else co["updates"]
        opt = torch.optim.AdamW(params, lr=co["lr"], weight_decay=0.0)
        first, ema = 1, {}
        if resume_path.exists():
            state = torch.load(resume_path, map_location=dev, weights_only=False)
            one.load_state_dict(state["model_state_dict"])
            opt.load_state_dict(state["optimizer"])
            rng.set_state(state["rng"].cpu())
            best["score"], first, ema = state["best"], state["update"] + 1, state["ema"]
            sigma = torch.tensor(state["sigma"], dtype=torch.float64, device=dev)
            event("cotrain_resumed", update=state["update"], best=best["score"])
        else:
            sigma = core_sigma(core_val())   # the history noise starts from this network's own error
            event("core_sigma", sigma=sigma.tolist(), source="the one network")
        kscale = one.target_std.to(torch.float64).view(n_mov, smax)
        h0 = max(1, round(co["horizon_start_s"] / three.dt))
        h1 = max(h0, round(co["horizon_max_s"] / three.dt))
        push_from = int(cfg["push_start"] * count)
        traj_every = co["traj_every"]
        for update in range(first, count + 1):
            frac = update / count
            for g in opt.param_groups:
                g["lr"] = co["lr"] * cosine(update, count, co["warmup"], 0.01)
            hb, okb, ab, bb, tb = body_window(core_rows(64 if smoke else co["core_batch"]))
            okb = drop_history(okb)
            terms = {"core": (((core_pred(hb, okb, ab, bb, sigma) - tb) / kscale[bb]).square() * core_masks[bb]).sum() / core_masks[bb].sum()}
            push = cfg["push_prob"] if update > push_from else 0.0
            terms["contact"] = contact_loss(sample(hit_windows, 64 if smoke else co["contact_batch"]), push)
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
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gnorm):
                event("cotrain_nonfinite_gradient", update=update)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            if update % (5 if smoke else cfg["evaluate_every"]) == 0 or update == count:
                sigma = core_sigma(core_val())   # the noise follows the network's current one-step error
                kept = push_stats["kept"] / max(1, push_stats["tried"])   # training pushes only (onestep adds its own)
                check("cotrain", update, total.detach(), {**onestep(), **{f"ema_{k}": v for k, v in ema.items()},
                                                          "push_kept_fraction": kept})
                push_stats.update(tried=0, kept=0)
                tmp = output / "state.pt.tmp"
                torch.save(dict(model_state_dict=one.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                                best=best["score"], update=update, ema=ema, sigma=sigma.tolist()), tmp)
                tmp.replace(resume_path)
        (output / "contact_done.json").write_text(json.dumps({"best": best["score"]}))

    # ---- 4. final validation of best.pt -------------------------------------------
    if not (output / "complete.json").exists():
        model_best, metadata = checkpoints.load_model(output / "best.pt", dev)
        with torch.no_grad():
            report = one_net.motion_evaluate(evaluate, model_best)(model_best, data, val, by_family=True)
        atomic_json(output / "validation.json", report)
        chosen = report.get("selected", report)   # the families that checkpoint selection used
        done = dict(complete=True, smoke=smoke, arch=args.arch, parameters=runtime["parameters"], selection="motion score",
                    test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                    checkpoint_sha256=file_sha256(output / "best.pt"), best_stage=metadata["stage"],
                    best_update=metadata["update"], validation_score=chosen["selection_score"], checkpoints_compared=checks["count"],
                    finite=report["finite_fraction"])
        atomic_json(output / "complete.json", done)
        event("complete", **done)
