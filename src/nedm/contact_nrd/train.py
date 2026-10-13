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

Data of the collection contract (contact_nrd/collection_contract.md) can add three things; without them the run is
the same as before. A moving body's ``channel_mask`` in system.json replaces the automatic rule (a channel is
predicted if its spread is at least 1 % of the largest channel of its type in the same body); it must give a pose
channel and its rate channel the same value. Contract data always carry it: data with families and a moving body
without it get a ``mask_warning`` log row (the automatic rule decides for that body). With ``family`` and
system.json ``families``, the checks use only validation episodes of the families with select: true, and the final
report adds the metrics of each family. With an ``evaluation`` block in system.json, the checks and the final report
use data.task_metrics and its selection score instead of data.free_metrics (the checks skip the rollout modes that
the score does not use). A family with ``"use": "core_only"`` (counterfactual branches, e.g. G1 walking states with
the floor switched off) trains the core only: load_data keeps its episodes apart, in a store cut after their
longest episode (no padding to the full record count), and only the core reads it: its windows enter the core's
linear fit, the core loss and the core validation report (core_validation.json by_use), and nothing else (not the
history noise, the channel-mask statistics, the collision network, the pair input statistics, the contact fit and
step_rms, the contact batches, the rollout starts, the checks or the final report). run_config.json counts its
episodes apart (core_only_episodes; training_episodes and validation_episodes are the other families). Families
without "use" serve every part, as before.

States are velocity-level: the rate channels are step-average rates (data.step_average_rates) and the pose of
each integrated body follows pose + step x rate (model.ContactNRD.assemble). The one-step losses train only the
rate channels; the rollout loss compares the whole state.

Fine-tuning (config "init_from", absent in train.json; NRD data rounds, configs/so101_policy/nrd_round/; ported from
the G1 policy study's train.py): the run starts from a trained run's checkpoint: its weights AND its data-fitted maps
(input and output standardisation of the three networks, the ridge-fitted linear paths of the core and the contact
network), its channel masks, its history noise sigma and its loss scales (contact_scale.json of that run). Nothing is
refitted on the new data: the core's linear fit, the collision network's input statistics and the contact network's
input statistics and linear fit are skipped (the core and contact linear paths go on training by gradient in
co-training, as in a full run). The config's model config (model step, history, network, integrated kinds, derived
and relative channels, contact mask, pair preview) must equal the checkpoint's, and the data must have the
checkpoint's bodies and pairs. The collision-only stage runs collision_updates updates (default 0: skipped; lr
collision_lr, warmup collision_warmup). Co-training runs cotrain.updates updates at cotrain.lr with cotrain.warmup, as
in a full run, with every option of the config (e.g. collision_cotrain trains the collision network in it).
check_first (default true) checks the starting model before the first update (stage "init", update 0), so best.pt
is the starting model unless a later check beats it.
    "init_from": "<run dir>"   or   {"run": "<run dir>", "checkpoint": "best.pt", "collision_updates": 0,
                                     "collision_lr": 1e-4, "collision_warmup": 0, "check_first": true}

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

from nedm.contact_nrd.data import (add_derived, add_relative, atomic_json, dilate, evaluate, file_rows, file_sha256, load_data,
                                   selected_episodes, step_average_rates, window_labels)
from nedm.contact_nrd.model import ContactNRD, KinematicChain, load_model, model_config, relative_channels

REPO_ROOT = Path(__file__).resolve().parents[3]
# Data rules by channel type: (state std floor, difference std floor, contact-loss kappa).
TYPE_RULES = {
    "position": (0.01, 1e-4, 1e-2), "angle": (0.01, 1e-4, 1e-2), "quaternion": (0.01, 1e-4, 1e-2),
    "velocity": (0.01, 1e-3, 5e-4), "angular_rate": (0.1, 1e-2, 5e-4), "angular_velocity": (0.1, 1e-2, 3e-3),
    # derived channels (config "derived_state"; model.KinematicChain): inputs only, never predicted
    "derived_position": (0.01, 1e-4, 1e-2), "derived_rotation": (0.01, 1e-4, 1e-2),
    "derived_velocity": (0.01, 1e-3, 5e-4), "derived_angular_velocity": (0.1, 1e-2, 3e-3),
    "relative_position": (0.01, 1e-4, 1e-2), "relative_rotation": (0.01, 1e-4, 1e-2), "relative_velocity": (0.01, 1e-3, 5e-4),
}


def cosine(update, count, warm, floor):
    return min(1.0, update / warm if warm else 1.0) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * update / count)))


def ridge(design, target, lam):
    gram = design.T @ design + lam * len(design) * torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
    return torch.linalg.solve(gram, design.T @ target)


INIT_DEFAULTS = dict(checkpoint="best.pt", collision_updates=0, collision_lr=1e-4, collision_warmup=0, check_first=True)


def init_spec(cfg):
    """The config's "init_from" block with its defaults (module docstring), or None (a run from scratch). A string
    is the run directory. Unknown keys raise (nothing is refitted on the new data: there is no refit option)."""
    spec = cfg.get("init_from")
    if not spec:
        return None
    spec = {"run": spec} if isinstance(spec, str) else dict(spec)
    unknown = set(spec) - set(INIT_DEFAULTS) - {"run"}
    if "run" not in spec or unknown:
        raise ValueError(f"init_from needs a run directory ('run') and takes only {sorted(INIT_DEFAULTS)}; got {sorted(spec)}")
    out = {**INIT_DEFAULTS, **spec}
    out["run"] = str(Path(out["run"]).expanduser().resolve())
    return out


def same_structure(a, b):
    """Differences between two system.json in what the model's shape depends on (moving bodies, channels, actions,
    pairs, steps) -> list of keys (empty: the same)."""
    keys = ("name", "kind", "moving", "state_dim", "channels", "channel_types", "action_dim")
    diffs = [k for k in ("record_step_s", "pairs") if a.get(k) != b.get(k)]
    ba, bb = a["bodies"], b["bodies"]
    if len(ba) != len(bb):
        return diffs + ["bodies"]
    for n, (x, y) in enumerate(zip(ba, bb)):
        diffs += [f"bodies[{n}].{k}" for k in keys if x.get(k) != y.get(k)]
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="data directory (system.json, unified_data.npz)")
    parser.add_argument("--output", type=Path, required=True, help="run directory (an existing run resumes)")
    parser.add_argument("--seed", type=int, help="overrides the config seed")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/contact_nrd/train.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="a few updates of every stage (pipeline check)")
    parser.add_argument("--deterministic", action="store_true", help="repeatable GPU runs on one machine (slower)")
    parser.add_argument("--trace-draws", type=Path, help="tests: write {component: episodes it drew windows from} "
                                                         "to this JSON file (the eligibility rules of core-only families)")
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = json.loads(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if int(cfg["history"]) < 2:
        raise ValueError("history must be >= 2")
    init = init_spec(cfg)   # fine-tuning from a trained run (module docstring), or None
    job = {"job_id": os.environ["SLURM_JOB_ID"]} if os.environ.get("SLURM_JOB_ID") else {}
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    dev, smoke = args.device, args.smoke
    # Core-only families ("use": "core_only") are not in states, actions, ... at all: load_data puts them in a short
    # store that only the core reads (see core_rows); every other part sees only the arrays below.
    data = load_data(args.data, dev, model_step_s=cfg["model_step_s"], core_only="separate")
    data = step_average_rates(data, set(cfg["integrate_kinds"]))
    # Optional derived channels (config "derived_state": a KinematicChain spec, or the path of one relative to the repo
    # root; absent in train.json = none): forward kinematics of an articulated body (e.g. the SO101 grasp point) as
    # extra state channels that every network sees and that the step recomputes from the new joint state.
    derived = cfg.get("derived_state")
    if isinstance(derived, str):
        derived = json.loads((REPO_ROOT / derived).read_text())
    data = add_derived(data, derived)
    # Optional relative channels (config "relative_state": {"body": rigid body, "frame": the derived body}; absent = none):
    # the rigid body's pose and velocity in the derived point's frame (model.relative_channels), inputs only.
    relative = cfg.get("relative_state")
    data = add_relative(data, relative)
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
    # In systems with actions, windows start on a command boundary (the command changes every model step);
    # systems without actions use every record.
    acting = any(int(b.get("action_dim", 0)) for b in moving_b)

    def windows_of(contacts_, lengths_, splits_):
        """Labels [N, W, P] (contact in the step or the step before), steps just after a contact [N, W, P], bodies
        touched in a window [N, W, D], training and validation windows [N, W] of one set of episodes."""
        labels_ = window_labels(contacts_, stride)
        # a step-average rate spans the previous model step: a contact in (r - s, r] still changes window r
        spread = torch.zeros_like(labels_)
        spread[:, stride:] = labels_[:, :-stride]
        labels_ = labels_ | spread
        nw = labels_.shape[1]
        previous = torch.cat((torch.zeros_like(labels_[:, :1]), window_labels(contacts_, 1)[:, : nw - 1]), 1)
        after_ = previous & ~labels_
        grid = ((torch.arange(nw, device=dev) % stride == 0) if acting else torch.ones(nw, dtype=torch.bool, device=dev))[None, :]
        valid = (torch.arange(nw, device=dev)[None, :] < (lengths_ - stride)[:, None]) & grid
        touched = labels_ | after_
        involved_ = torch.zeros(len(labels_), nw, n_mov, dtype=torch.bool, device=dev)
        for p, (i, j) in enumerate(pairs):
            involved_[..., slot[i]] |= touched[..., p]
            if j in slot:
                involved_[..., slot[j]] |= touched[..., p]
        return labels_, after_, involved_, valid & (splits_ == 0)[:, None], valid & (splits_ == 1)[:, None]

    labels, after, involved, train, vwin = windows_of(contacts, lengths, splits)          # labels [N, W, P]
    n_win = labels.shape[1]
    store = data.get("core_store")   # core-only episodes; a core window of row e of the store has episode n_ep + e
    c_states = c_actions = None
    if store is not None:
        c_states, c_actions = store["states"], store["actions"]
        _, _, c_involved, c_train, c_vwin = windows_of(store["contacts"], store["lengths"], store["splits"])
        shift = torch.tensor([n_ep, 0], device=dev)
    drawn = {} if args.trace_draws else None

    def trace(component, episodes):
        """--trace-draws: remember the episodes (rows of unified_data.npz) of the windows that a component draws (tests)."""
        if drawn is not None and len(episodes):
            e = torch.unique(episodes)
            if "file_index" in data:
                main = e < n_ep
                e = torch.cat((data["file_index"][e[main]], store["file_index"][e[~main] - n_ep] if store is not None else e[:0]))
            drawn.setdefault(component, set()).update(e.tolist())
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

    def hist_of(windows, src=None):
        src = states if src is None else src
        raw = windows[:, 1, None] + torch.arange(1 - K, 1, device=dev) * stride
        ok = raw >= 0
        h = src[windows[:, 0, None], raw.clamp_min(0)] * ok[..., None, None]
        return h, ok

    def act_of(windows):
        return actions[windows[:, 0], windows[:, 1]]

    def drop_history(ok):
        # step-average rates: a one-entry history occurs only at an episode start (recorded rates), never later
        keep = torch.randint(2, K + 1, (len(ok),), device=dev, generator=rng)
        return ok & (torch.arange(K, device=dev)[None, :] >= K - keep[:, None])

    # ---- channel masks (constant channels by type, per body; padding off) --------------
    sw = sample(torch.nonzero(train), 100000)
    trace("channel_masks", sw[:, 0])
    s0 = states[sw[:, 0], sw[:, 1]]                                                    # [n, D, S]
    span = s0.quantile(0.999, dim=0) - s0.quantile(0.001, dim=0)                       # [D, S]
    masks = torch.zeros(n_mov, smax, dtype=torch.float64, device=dev)
    for d, ts in enumerate(types):
        for c, t in enumerate(ts):
            same = [cc for cc, tt in enumerate(ts) if tt == t]
            masks[d, c] = float(span[d, c] >= 0.01 * span[d, same].max())
    automatic = masks.clone()
    explicit = [d for d, b in enumerate(moving_b) if "channel_mask" in b]
    for d in explicit:   # the data's own mask (e.g. a robot's base height, small next to its travel)
        cm = [float(x) for x in moving_b[d]["channel_mask"]]
        if len(cm) != len(types[d]) or set(cm) - {0.0, 1.0}:
            raise ValueError(f"{moving_b[d]['name']}: channel_mask needs one 0/1 per channel")
        masks[d] = 0.0
        masks[d, : len(cm)] = torch.tensor(cm, dtype=torch.float64, device=dev)

    mc = model_config(data["dt"], K, cfg["network"], cfg["integrate_kinds"], derived, cfg.get("contact_mask"),
                      bool(cfg.get("pair_preview", False)), relative, bool(cfg.get("aux_geometry")))
    packet, init_notes = None, {}
    if init is not None:   # the checkpoint's design, masks and weights; the data must have its bodies and pairs
        packet = torch.load(Path(init["run"]) / init["checkpoint"], map_location=dev, weights_only=False)
        if not (Path(init["run"]) / "contact_scale.json").exists():
            raise ValueError(f"init_from: {init['run']} has no contact_scale.json (the loss scales of that run)")
        if packet["model_config"] != mc:
            raise ValueError(f"init_from: the config gives the model config {mc}, the checkpoint has {packet['model_config']}")
        diffs = same_structure(packet["system"], system)
        if diffs:
            raise ValueError(f"init_from: the data's system differs from the checkpoint's in {diffs}")
        ck = torch.tensor(packet["masks"], dtype=torch.float64, device=dev)
        if not torch.equal(ck, masks):
            init_notes["data_masks"] = masks.tolist()
        masks = ck
        if packet.get("sigma") is None:
            raise ValueError("init_from: the checkpoint has no history-noise sigma")
    model = ContactNRD(mc, system, masks.tolist()).to(dev)
    if packet is not None:
        model.load_state_dict(packet["model_state_dict"])
    # The integrator pairs pose and rate channels by order, so their masks must agree; a quaternion's mask is
    # [any angular axis, x, y, z] of its angular velocity mask (all on, all off, or a planar body's [1, 0, 0, 1]).
    unpaired = []
    for n in torch.nonzero(model.integ_body).flatten().tolist():
        pc, qc, vc, wc = (model.chan[n, k][model.chan[n, k] >= 0] for k in range(4))
        m = masks[n]
        agree = torch.equal(m[pc], m[vc])
        if len(qc) == 4 and len(wc) == 3:
            agree = agree and torch.equal(m[qc], torch.cat((m[wc].amax()[None], m[wc])))
        if not agree:
            unpaired.append(moving_b[n]["name"])
            if n in explicit:
                raise ValueError(f"{moving_b[n]['name']}: channel_mask must give a pose channel and its rate channel "
                                 "the same value (quaternion: [any angular axis, x, y, z])")
    # Channels that are trained (core and contact losses): integrated pose channels are computed, not predicted.
    core_masks = masks.clone()
    for b in torch.nonzero(model.integ_body).flatten().tolist():
        for k in (0, 1):
            cols = model.chan[b, k][model.chan[b, k] >= 0]
            core_masks[b, cols] = 0.0
    derived_cols = torch.tensor([[t.startswith(("derived_", "relative_")) for t in ts] + [False] * (smax - len(ts))
                                 for ts in types], device=dev)   # input-only channels (derived and relative)
    core_masks[derived_cols] = 0.0   # derived channels are computed from the joint state, not predicted

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
        for n, ts in enumerate(types):   # derived pose channels from the noise of the derived velocities
            for pose_t, rate_t in (("derived_position", "derived_velocity"), ("derived_rotation", "derived_angular_velocity"),
                                   ("relative_position", "relative_velocity"), ("relative_rotation", "angular_velocity")):
                pc = [c for c, t in enumerate(ts) if t == pose_t]
                vc = [c for c, t in enumerate(ts) if t == rate_t]
                if pc and vc:
                    sig[n, pc] = model.dt * sig[n, vc].mean() * masks[n, pc]
        return sig

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sigma = None   # history-noise scale [D, S]: the core's one-step error (updated at every check)
    if packet is not None:   # fine-tuning starts from the checkpoint's noise
        sigma = torch.tensor(packet["sigma"], dtype=torch.float64, device=dev)

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
    if family is not None:
        runtime.update(selection_families=[f["name"] for f in system["families"] if f["select"]],
                       check_episodes_used=len(val_check))
    if store is not None:   # training_episodes and validation_episodes above count only the other families
        runtime.update(core_only_families=[f["name"] for f in system["families"] if f.get("use") == "core_only"],
                       core_only_episodes={"train": int((store["splits"] == 0).sum()),
                                           "validation": int((store["splits"] == 1).sum())})
    if packet is not None:
        ck_path = Path(init["run"]) / init["checkpoint"]
        runtime.update(init_from=dict(init, checkpoint_sha256=file_sha256(ck_path), source_stage=packet.get("stage"),
                                      source_update=packet.get("update"),
                                      contact_scale_sha256=file_sha256(Path(init["run"]) / "contact_scale.json"),
                                      source_data_sha256=(packet.get("runtime") or {}).get("data_sha256")))
    if not (output / "run_config.json").exists():
        atomic_json(output / "run_config.json", runtime)
    else:
        event("resume", **job)
    if init_notes:
        event("init_from_masks", note="the checkpoint's channel masks are used; the data's differ", **init_notes)
    event("masks", masks=masks.tolist(), **({"explicit": [moving_b[d]["name"] for d in explicit],
                                             "automatic": automatic.tolist()} if explicit else {}))
    if unpaired:   # automatic masks only (explicit ones raise above)
        event("mask_warning", bodies=unpaired, note="a pose channel and its rate channel have different masks")
    unmasked = [b["name"] for d, b in enumerate(moving_b) if d not in explicit]
    if family is not None and unmasked:   # contract data (with families) always carry channel_mask
        event("mask_warning", bodies=unmasked, note="data with families but no channel_mask for these bodies: the "
              "automatic rule decides (collection_contract.md requires channel_mask)")

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
        trace("checks", val_check)
        with torch.no_grad():
            report = evaluate(model, data, val_check, score_only=True)
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
                  moved_pos_median_mm=1000 * report["moved_target_pos_m"]["median"], events_ok=report["event_ok_fraction"],
                  finite=report["finite_fraction"], **(extra or {}))
        if was:
            model.train()

    # ---- 1. core -------------------------------------------------------------------
    core = model.core
    nk = len(model.kind_names)
    co = cfg["cotrain"]
    # Free-step windows of the core (every free_phase_step records), grouped per body kind into core_bins bins of
    # log |rate change|; a batch draws from each bin in proportion to the square root of its size. A core window is
    # (episode, record, body); episodes n_ep and up are rows of the core-only store (training windows of both).

    def phase_keep(width):
        return (torch.arange(width, device=dev) % cfg["free_phase_step"] == 0)[None, :]

    def at(src, src_core, ep, rec, body):
        """src[ep, rec, body]; rows with ep >= n_ep read src_core[ep - n_ep, rec, body] (the core-only store)."""
        if store is None:
            return src[ep, rec, body]
        main = ep < n_ep
        out = src.new_empty((len(ep), *src.shape[3:]))
        out[main] = src[ep[main], rec[main], body[main]]
        out[~main] = src_core[ep[~main] - n_ep, rec[~main], body[~main]]
        return out

    def core_rows(tr):
        """History [n, K, 1, S] of each row's own body, its flags, action [n, 1, A] and target (next state - current,
        masked) of core windows tr = (episode, record, body), from the arrays or the core-only store."""
        if store is None:
            h, ok = hist_of(tr[:, :2])
        else:
            main = tr[:, 0] < n_ep
            h = states.new_empty((len(tr), K, n_mov, smax))
            ok = torch.empty((len(tr), K), dtype=torch.bool, device=dev)
            h[main], ok[main] = hist_of(tr[main, :2])
            h[~main], ok[~main] = hist_of(tr[~main, :2] - shift, c_states)
        h = h[torch.arange(len(tr), device=dev), :, tr[:, 2]][:, :, None]                 # [n, K, 1, S]
        a = at(actions, c_actions, tr[:, 0], tr[:, 1], tr[:, 2])[:, None]                 # [n, 1, A]
        target = (at(states, c_states, tr[:, 0], tr[:, 1] + stride, tr[:, 2]) - h[:, -1, 0]) * masks[tr[:, 2]]
        return h, ok, a, target

    groups, weights = [], []
    triples_by_kind = {}
    for b in range(n_mov):
        w = torch.nonzero(train & phase_keep(n_win) & ~involved[..., b])
        if store is not None:
            w = torch.cat((w, torch.nonzero(c_train & phase_keep(c_train.shape[1]) & ~c_involved[..., b]) + shift))
        triples_by_kind.setdefault(int(model.body_kind[b]), []).append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
    for kk, parts in triples_by_kind.items():
        triples = torch.cat(parts)
        b0 = triples[:, 2]
        dch = torch.cat([(at(states, c_states, c[:, 0], c[:, 1] + stride, c[:, 2])
                          - at(states, c_states, c[:, 0], c[:, 1], c[:, 2])) for c in triples.split(1 << 20)])
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
        trace("core", tr[:, 0])
        h, ok, a, target = core_rows(tr)
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
        out = core(cur, diff, a, present, has, kind[:, None], lag_ok, fixed, kind, model.core_smask[body][:, None], model.amask[body][:, None])
        out = out.reshape(len(h), nk, smax)[torch.arange(len(h), device=dev), kind]
        return out * masks[body]

    if (output / "core.pt").exists():
        load_stage(output / "core.pt")
        event("core_loaded")
    elif packet is not None:   # the checkpoint's core, input and output standardisation and linear path (no refit)
        event("init_from", run=init["run"], checkpoint=init["checkpoint"], source_stage=packet.get("stage"),
              source_update=packet.get("update"), sigma="the checkpoint's", note="no linear fit, no refit")
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
                m_s, m_a = model.core_smask[bsel[0]], model.amask[bsel[0]]   # the input mask that run_core uses
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
    # Contact-free validation steps of the core (its one-step error sets the history noise; with core-only episodes
    # their windows are in the core report, but the noise comes from the other windows: see noise_rows).
    with torch.no_grad():
        pv = []
        for b in range(n_mov):
            w = torch.nonzero(vwin & ~involved[..., b])
            if store is not None:
                w = torch.cat((w, torch.nonzero(c_vwin & ~c_involved[..., b]) + shift))
            w = w[torch.randperm(len(w), device=dev, generator=rng)[:100000]]
            pv.append(torch.cat((w, torch.full_like(w[:, :1], b)), -1))
        pv = torch.cat(pv)
        trace("core_validation", pv[:, 0])
        pv_h, pv_ok, pv_a, pv_t = core_rows(pv)
        pv_b = pv[:, 2]
        # The history noise is an input noise of the collision and contact training, so core-only windows do not set
        # it (a body without other validation windows uses all of its own).
        noise_rows = [pv_b == b for b in range(n_mov)]
        if store is not None:
            from_core = pv[:, 0] >= n_ep
            noise_rows = [r & ~from_core if bool((r & ~from_core).any()) else r for r in noise_rows]
            event("history_noise_windows", bodies={moving_b[b]["name"]: {"used": int(noise_rows[b].sum()),
                                                                          "core_only_left_out": int((noise_rows[b] != (pv_b == b)).sum())}
                                                   for b in range(n_mov)})

    def core_val():
        with torch.no_grad():
            return torch.cat([core_pred(h, o, a, b) for h, o, a, b in
                              zip(pv_h.split(16384), pv_ok.split(16384), pv_a.split(16384), pv_b.split(16384))]) - pv_t

    # Optional history-noise rule (config "history_noise", absent in train.json = the released rule): the noise is
    # multiplied by "scale" and capped per channel at "cap_step_rms" x the RMS of the one-step change on training
    # steps with one active pair (for systems whose contact-free steps are much harder than their contact steps).
    # With "floor_step_rms" the noise of each rate channel is at least that fraction of the same RMS (for bodies whose
    # free motion is far easier than their contact motion, e.g. a cube that only flies when it is free: the pair
    # networks must still see history errors of the size that their own contact steps make).
    noise_rule = cfg.get("history_noise") or {}
    noise_scale, noise_cap = float(noise_rule.get("scale", 1.0)), noise_rule.get("cap_step_rms")
    noise_floor = noise_rule.get("floor_step_rms")
    if set(noise_rule) - {"scale", "cap_step_rms", "floor_step_rms", "skip_first_step", "pose_shift", "pose_shift_for", "rate_kick"}:
        raise ValueError(f"history_noise takes scale, cap_step_rms, floor_step_rms, skip_first_step, pose_shift and rate_kick, not {noise_rule}")
    pose_shift = noise_rule.get("pose_shift")   # see shift_poses
    shift_for = set(noise_rule.get("pose_shift_for", ["collision", "contact"]))   # the networks whose examples it moves
    if shift_for - {"collision", "contact"}:
        raise ValueError(f"history_noise pose_shift_for takes collision and/or contact, not {shift_for}")
    rate_kick = noise_rule.get("rate_kick")     # see kick_rates
    if noise_rule.get("skip_first_step"):   # first-step windows (one history entry, recorded rates) follow another rule
        full_hist = pv_ok.sum(-1) >= 2
        noise_rows = [r & full_hist if bool((r & full_hist).any()) else r for r in noise_rows]
    if noise_cap is not None or noise_floor is not None:
        with torch.no_grad():
            one = torch.nonzero(train & (labels.sum(-1) == 1))
            one = one[torch.linspace(0, len(one) - 1, min(len(one), 400000), device=dev).round().long()]
            trace("noise_cap", one[:, 0])
            on = labels[one[:, 0], one[:, 1]]
            change = (states[one[:, 0], one[:, 1] + stride] - states[one[:, 0], one[:, 1]]) * masks
            one_rms = torch.full((n_mov, smax), float("inf"), dtype=torch.float64, device=dev)
            for b in range(n_mov):
                hit = torch.zeros(len(one), dtype=torch.bool, device=dev)
                for p, (i, j) in enumerate(pairs):
                    if slot[i] == b or slot.get(j, -1) == b:
                        hit |= on[:, p]
                if hit.any():
                    one_rms[b] = change[hit, b].square().mean(0).sqrt()
            cap_rms = float(noise_cap) * one_rms if noise_cap is not None else None
            floor_rms = None
            if noise_floor is not None:   # rate channels only; the pose noise follows from the rates (pose_noise)
                rate_cols = torch.zeros(n_mov, smax, dtype=torch.bool, device=dev)
                for d, ts in enumerate(types):
                    for c, t in enumerate(ts):
                        rate_cols[d, c] = t in ("velocity", "angular_rate", "angular_velocity", "derived_velocity",
                                                "derived_angular_velocity", "relative_velocity")
                floor_rms = torch.where(rate_cols & torch.isfinite(one_rms), float(noise_floor) * one_rms, 0.0)
        event("history_noise_rule", scale=noise_scale, cap_step_rms=noise_cap, floor_step_rms=noise_floor,
              cap=None if cap_rms is None else torch.where(torch.isfinite(cap_rms), cap_rms, -1.0).tolist(),
              floor=None if floor_rms is None else floor_rms.tolist())

    def core_sigma(err):
        rms = torch.stack([err[noise_rows[b]].square().mean(0).sqrt() for b in range(n_mov)])
        rms = torch.where(derived_cols, 0.0, rms)   # derived channels: no core error (only the floor rule gives them noise)
        if noise_cap is not None:
            rms = torch.minimum(rms, cap_rms)
        if noise_floor is not None:
            rms = torch.maximum(rms, floor_rms)
        return pose_noise(rms * masks) * noise_scale if noise_rule else pose_noise(rms * masks)

    if not (output / "core.pt").exists():
        err = core_val()   # co-training starts from the linear core; the noise follows that core's error
        if packet is None:
            sigma = core_sigma(err)
            event("core_sigma", sigma=sigma.tolist(), source="linear core (co-training)")
        else:   # fine-tuning keeps the checkpoint's noise until the first check (as after a check of a full run)
            event("core_sigma", sigma=sigma.tolist(), source="init_from checkpoint", this_data=core_sigma(err).tolist())
        with torch.no_grad():
            report = {}
            for b in range(n_mov):
                e = err[pv_b == b].abs()
                report[moving_b[b]["name"]] = {"median": e.median(0).values.tolist(), "p95": torch.quantile(e.float(), 0.95, dim=0).tolist()}
            by_use = {}
            if store is not None:   # the same windows split into those of the other and of core-only families
                for use, part in (("task", ~from_core), ("core_only", from_core)):
                    by_use[use] = {}
                    for b in range(n_mov):
                        e = err[(pv_b == b) & part].abs()
                        if len(e):
                            by_use[use][moving_b[b]["name"]] = {"windows": len(e), "median": e.median(0).values.tolist(),
                                                                "p95": torch.quantile(e.float(), 0.95, dim=0).tolist(),
                                                                "rms": e.square().mean(0).sqrt().tolist()}
        atomic_json(output / "core_validation.json", {**report, **({"by_use": by_use} if by_use else {})})
        event("core_validation", **{k: v["median"] for k, v in report.items()})
        save(output / "core.pt", "core" if packet is None else "init_from", 0)
    for p in core.parameters():
        p.requires_grad_(False)
    if sigma is None:
        raise RuntimeError("core noise scale missing")

    # Optional training-only geometry heads (config "aux_geometry": {"fk": KinematicChain spec path, "body": rigid body,
    # "frame": articulated body, "pairs": [pair names], "weight": w, "generated": true}; absent = none). The heads of
    # the collision and contact networks (model.ContactNRD.geometry) learn the body's pose and slip in the frame
    # point's frame (model.relative_channels of forward kinematics; units 1 mm, 0.02 and 5 mm/s; Huber): in stage 2 on
    # the collision batches of the given pairs, in stage 3 on contact windows of those pairs (and on the collision
    # batches with collision_cotrain). With "generated", as many again come from generated poses near the frame point
    # (recorded joint states with N(0, 0.15 rad) on the chain joints and a uniform jaw angle, recorded rates, the body
    # placed within 40 mm of the point, upright or aligned with the gripper, history integrated backward at constant
    # rate). Only the heads and the shared trunks learn from them, never the switch or contact outputs. Forward
    # kinematics enters only the loss: the state and the step stay as they are.
    geo_cfg = cfg.get("aux_geometry")
    if geo_cfg:
        geo_kin = KinematicChain(json.loads((REPO_ROOT / geo_cfg["fk"]).read_text())).to(dev)
        names_ = [b["name"] for b in moving_b]
        geo_b, geo_f = names_.index(geo_cfg["body"]), names_.index(geo_cfg["frame"])
        ft_, bt_ = types[geo_f], types[geo_b]
        geo_angle = [c for c, t in enumerate(ft_) if t == "angle"]
        geo_rate = [c for c, t in enumerate(ft_) if t == "angular_rate"]
        geo_q, geo_qd = geo_angle[: geo_kin.n], geo_rate[: geo_kin.n]
        geo_p = [c for c, t in enumerate(bt_) if t == "position"]
        geo_o = [c for c, t in enumerate(bt_) if t == "quaternion"]
        geo_v = [c for c, t in enumerate(bt_) if t == "velocity"]
        geo_w = [c for c, t in enumerate(bt_) if t == "angular_velocity"]
        pnames_ = [m.get("name") for m in system.get("pair_metadata", [])]
        geo_pairs = torch.tensor([pnames_.index(n) for n in geo_cfg["pairs"]], device=dev)
        geo_unit = torch.tensor([1e-3] * 3 + [0.02] * 6 + [5e-3] * 3, dtype=torch.float64, device=dev)
        geo_weight, geo_gen = float(geo_cfg.get("weight", 1.0)), bool(geo_cfg.get("generated", True))

        def geo_target(x):
            """[n, D, S] -> [n, 12] (units of geo_unit)."""
            d_ = geo_kin(x[:, geo_f, geo_q], x[:, geo_f, geo_qd], model.dt)
            return relative_channels(d_, x[:, geo_b, geo_p], x[:, geo_b, geo_o], x[:, geo_b, geo_v]) / geo_unit

        geo_src = torch.nonzero(train)
        with torch.no_grad():   # pool of training windows with the body near the frame point; ranges for generated poses
            cand = sample(geo_src, 200000)
            xs = states[cand[:, 0], cand[:, 1]]
            near = (geo_target(xs)[:, :3] * 1e-3).norm(dim=-1) < 0.06
            geo_pool = cand[near]
            ang_lo, ang_hi = xs[:, geo_f, geo_angle].quantile(0.001, 0), xs[:, geo_f, geo_angle].quantile(0.999, 0)
        trace("aux_geometry_pool", geo_pool[:, 0])
        event("aux_geometry", pool_windows=len(geo_pool), pairs=geo_cfg["pairs"], generated=geo_gen)

        def _quat_of(r):
            """Rotation matrices [n, 3, 3] -> unit quaternions [n, 4] (w first, w >= 0)."""
            m = r
            w = torch.sqrt((1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]).clamp_min(1e-12)) / 2
            x_ = torch.sqrt((1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp_min(0)) / 2 * torch.sign(m[:, 2, 1] - m[:, 1, 2])
            y_ = torch.sqrt((1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2]).clamp_min(0)) / 2 * torch.sign(m[:, 0, 2] - m[:, 2, 0])
            z_ = torch.sqrt((1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]).clamp_min(0)) / 2 * torch.sign(m[:, 1, 0] - m[:, 0, 1])
            q = torch.stack((w, x_, y_, z_), -1)
            return q / q.norm(dim=-1, keepdim=True)

        def _rot(axis, ang):
            """Rotation matrices about unit axes [n, 3] by angles [n]."""
            k = torch.zeros(len(axis), 3, 3, dtype=axis.dtype, device=dev)
            k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
            k = k - k.transpose(1, 2)
            s_, c_ = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
            return torch.eye(3, dtype=axis.dtype, device=dev) + s_ * k + (1 - c_) * (k @ k)

        def geo_generated(n):
            """n generated (history, ok, action) inputs with the body near the frame point (see aux_geometry)."""
            w_ = sample(geo_pool, n)
            x = states[w_[:, 0], w_[:, 1]].clone()
            src = sample(geo_src, n)
            xr = states[src[:, 0], src[:, 1]]
            ang = x[:, geo_f, geo_angle]
            ang[:, : geo_kin.n] = ang[:, : geo_kin.n] + torch.randn_like(ang[:, : geo_kin.n]) * 0.15
            ang[:, geo_kin.n:] = ang_lo[geo_kin.n:] + torch.rand_like(ang[:, geo_kin.n:]) * (ang_hi - ang_lo)[geo_kin.n:]
            ang = torch.maximum(torch.minimum(ang, ang_hi), ang_lo)
            x[:, geo_f, geo_angle] = ang
            x[:, geo_f, geo_rate] = xr[:, geo_f, geo_rate]
            x[:, geo_b, geo_v], x[:, geo_b, geo_w] = xr[:, geo_b, geo_v], xr[:, geo_b, geo_w]
            d_ = geo_kin(x[:, geo_f, geo_q], x[:, geo_f, geo_qd], model.dt)
            gp, c0, c1 = d_[:, 0:3], d_[:, 3:6], d_[:, 6:9]
            rg = torch.stack((c0, c1, torch.cross(c0, c1, dim=-1)), -1)
            off = torch.stack((torch.rand(n, device=dev) * 0.08 - 0.04, torch.rand(n, device=dev) * 0.08 - 0.04,
                               torch.rand(n, device=dev) * 0.07 - 0.045), -1).to(x.dtype)
            x[:, geo_b, geo_p] = gp + (rg @ off[..., None])[..., 0]
            up = torch.rand(n, device=dev) < 0.7
            zaxis = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=dev).expand(n, 3)
            yaw = _rot(zaxis, torch.rand(n, device=dev, dtype=x.dtype) * 2 * math.pi)
            tilt_axis = torch.cat((torch.randn(n, 2, device=dev, dtype=x.dtype), torch.zeros(n, 1, device=dev, dtype=x.dtype)), -1)
            tilt_axis = tilt_axis / tilt_axis.norm(dim=-1, keepdim=True)
            r_up = _rot(tilt_axis, torch.rand(n, device=dev, dtype=x.dtype) * math.radians(3)) @ yaw
            quarter = torch.randint(0, 4, (n,), device=dev).to(x.dtype) * (math.pi / 2) + (torch.rand(n, device=dev, dtype=x.dtype) - 0.5) * math.radians(20)
            r_al = rg @ _rot(zaxis, quarter)
            x[:, geo_b, geo_o] = _quat_of(torch.where(up[:, None, None], r_up, r_al))
            hist = x[:, None].repeat(1, K, 1, 1)
            for k in range(1, K):   # backward at constant rate (step averages stay consistent)
                y = hist[:, K - 1 - k + 1].clone()
                prev = y.clone()
                prev[:, geo_f, geo_angle] = y[:, geo_f, geo_angle] - model.dt * y[:, geo_f, geo_rate]
                prev[:, geo_b, geo_p] = y[:, geo_b, geo_p] - model.dt * y[:, geo_b, geo_v]
                half = -0.5 * model.dt * y[:, geo_b, geo_w]
                a_ = half.norm(dim=-1, keepdim=True)
                dq = torch.cat((torch.cos(a_), torch.where(a_ > 1e-12, torch.sin(a_) / a_.clamp_min(1e-12), torch.ones_like(a_)) * half), -1)
                prev[:, geo_b, geo_o] = model._qmul(dq, y[:, geo_b, geo_o])
                hist[:, K - 1 - k] = prev
            hist = model.finish(hist.reshape(n * K, n_mov, smax)).reshape(n, K, n_mov, smax)
            ok_ = torch.ones(n, K, dtype=torch.bool, device=dev)
            act = torch.zeros(n, n_mov, actions.shape[-1], dtype=x.dtype, device=dev)
            act[:, geo_f, : len(geo_angle)] = ang
            return hist, ok_, act

        def geo_loss(net, h, ok, act):
            """Huber loss of a geometry head on inputs (history noise as in training; random jaw pair)."""
            ok = drop_history(ok)
            cur, diff, lag_ok = model.code(h, ok, sigma)
            pairs_ = geo_pairs[torch.randint(len(geo_pairs), (len(h),), device=dev, generator=rng)]
            pred = model.geometry(net, cur, diff, act, lag_ok, torch.arange(len(h), device=dev), pairs_)
            return F.huber_loss(pred, geo_target(cur), delta=1.0)

        def geo_term(net, windows):
            """Geometry loss on recorded windows (and as many generated inputs with "generated")."""
            h, ok = hist_of(windows)
            loss = geo_loss(net, h, ok, act_of(windows))
            if geo_gen:
                loss = loss + geo_loss(net, *geo_generated(len(windows)))
            return loss
        geo_windows = torch.nonzero(train & labels[..., geo_pairs].any(-1))   # recorded steps with a jaw pair on

    # ---- 2. collision -------------------------------------------------------------
    coll = model.collision
    near_rec = max(1, round(cfg["near_negative_s"] / data["record_dt"]))
    near_pair = dilate(labels, near_rec) & train[..., None]

    def pair_raw(windows, pairs_):
        h, ok = hist_of(windows)
        cur, diff, lag_ok = model.code(h, ok)
        rows = torch.arange(len(windows), device=dev)
        return model.pair_inputs(cur, diff, act_of(windows), lag_ok, rows, pairs_)

    def fit_inputs(net, pools, component):
        """Input standardisation of a pair network, per pair and role position."""
        with torch.no_grad():
            for p, wins in enumerate(pools):
                wins = wins[torch.randperm(len(wins), device=dev, generator=rng)[:50000]]
                trace(component, wins[:, 0])
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
    # Optional collision sampling (config "collision_sampling": "sqrt_families", absent = the released rule): the
    # training contact and near-contact steps of each pair are drawn by family, each family in proportion to the square
    # root of its count, so that rare contact geometries (a fingertip landing on the cube's top in a near miss, a single
    # jaw against the cube on the table) are not drowned by the holds of normal grasps. Needs a family array.
    coll_sampling = cfg.get("collision_sampling")
    if coll_sampling not in (None, "sqrt_families") or (coll_sampling and family is None):
        raise ValueError(f"collision_sampling takes sqrt_families (data with families), not {coll_sampling}")

    def by_family(pool):
        """[(windows of one family, weight)] of a pool of training windows (sqrt of the family counts)."""
        f = family[pool[:, 0]]
        groups = [pool[f == k] for k in torch.unique(f).tolist()]
        w = torch.tensor([math.sqrt(len(g)) for g in groups], dtype=torch.float64)
        return list(zip(groups, (w / w.sum()).tolist()))
    fam_strata = [(by_family(pos), by_family(neg)) if coll_sampling else None for pos, neg, _, _ in strata]

    def draw(pool, groups, n):
        if groups is None:
            return sample(pool, n)
        return torch.cat([sample(g, max(1, int(round(n * w)))) for g, w in groups])

    def collision_batch(n, split="train"):
        """Per pair: 1/3 contact steps, 1/3 near-contact steps, the rest any step."""
        out = []
        for p, (pos, neg, vpos, vneg) in enumerate(strata):
            fg = fam_strata[p] if split == "train" else None
            pos, neg, anyw = (pos, neg, any_train) if split == "train" else (vpos, vneg, any_val)
            m = max(3, n // n_pairs)
            parts = [draw(pos, fg and fg[0], m // 3)] if len(pos) else []
            parts += [draw(neg, fg and fg[1], m // 3)] if len(neg) else []
            parts.append(sample(anyw, max(0, m - sum(len(q) for q in parts))))
            wnd = torch.cat(parts)
            out.append(torch.cat((wnd, torch.full_like(wnd[:, :1], p)), -1))
        out = torch.cat(out)
        trace("collision" if split == "train" else "collision_validation", out[:, 0])
        return out

    pose_types = {"position", "angle"}
    shift_cols = torch.tensor([[t in pose_types for t in ts] + [False] * (smax - len(ts)) for ts in types], device=dev)
    quat_cols = [torch.tensor([c for c, t in enumerate(ts) if t == "quaternion"], dtype=torch.long, device=dev) for ts in types]

    def shift_poses(h, net):
        """history_noise "pose_shift" k (pair-network training only): move the whole history of every body by one
        random pose offset of k x its pose noise (positions and joint angles shifted, quaternions turned by one small
        world-frame rotation; derived channels recomputed), so that the switch and contact networks see the absolute
        drift of a rollout (a resting cube that is 1 mm high or tilted by a degree), not only noisy differences."""
        if not pose_shift or net not in shift_for:
            return h
        b, k = h.shape[:2]
        off = torch.randn(b, 1, n_mov, smax, dtype=h.dtype, device=dev, generator=None) * sigma * float(pose_shift)
        h = h + torch.where(shift_cols, off, 0.0)
        for n, qc in enumerate(quat_cols):
            if len(qc) == 4:
                half = 0.5 * torch.randn(b, 1, 3, dtype=h.dtype, device=dev) * (2 * sigma[n, qc[1:]] * float(pose_shift))
                ang = half.norm(dim=-1, keepdim=True)
                dq = torch.cat((torch.cos(ang), torch.where(ang > 1e-12, torch.sin(ang) / ang.clamp_min(1e-12), torch.ones_like(ang)) * half), -1)
                h = h.clone()
                h[:, :, n, qc] = model._qmul(dq.expand(b, k, 4), h[:, :, n, qc])
        return model.finish(h.reshape(b * k, n_mov, smax)).reshape(b, k, n_mov, smax)

    def kick_rates(h, ok):
        """history_noise "rate_kick" {"size": k, "mix": [lo, hi]} (contact examples only): the newest history state
        of every integrated body gets a rate error e ~ N(0, (m k sigma)^2) with a per-example log-uniform factor m
        (heavy tail, as rollout drift); its pose moves by dt e (quaternion: turned by dt e_w) and the derived channels
        follow. The target stays the recorded next state, so the contact network learns to remove a spurious slip of
        its own state within one step (a held cube whose velocity left the gripper's)."""
        if not rate_kick:
            return h
        b = h.shape[0]
        lo, hi = rate_kick.get("mix", [1.0, 1.0])
        m = torch.exp(torch.empty(b, 1, 1, dtype=h.dtype, device=dev).uniform_(math.log(lo), math.log(hi)))
        e = torch.randn(b, n_mov, smax, dtype=h.dtype, device=dev) * sigma * float(rate_kick["size"]) * m
        cur = h[:, -1].clone()
        for n in torch.nonzero(model.integ_body).flatten().tolist():
            pc, qc, vc, wc = (model.chan[n, k][model.chan[n, k] >= 0] for k in range(4))
            if len(vc):
                cur[:, n, vc] += e[:, n, vc]
                if len(pc):
                    cur[:, n, pc] += model.dt * e[:, n, vc]
            if len(wc) == 3:
                cur[:, n, wc] += e[:, n, wc]
                if len(qc) == 4:
                    half = 0.5 * model.dt * e[:, n, wc]
                    ang = half.norm(dim=-1, keepdim=True)
                    dq = torch.cat((torch.cos(ang), torch.where(ang > 1e-12, torch.sin(ang) / ang.clamp_min(1e-12), torch.ones_like(ang)) * half), -1)
                    cur[:, n, qc] = model._qmul(dq, cur[:, n, qc])
        h = h.clone()
        h[:, -1] = torch.where(ok[:, -1, None, None], model.finish(cur), h[:, -1])
        return h

    def collision_logits(picks, train_mode):
        h, ok = hist_of(picks[:, :2])
        if train_mode:
            ok = drop_history(ok)
            h = shift_poses(h, "collision")
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
        spec = cfg["collision"]
        if packet is None:
            fit_inputs(coll, pools, "collision_inputs")
            c_lr, c_warm, count = spec["lr"], spec["warmup"], 5 if smoke else spec["updates"]
        else:   # fine-tuning: the checkpoint's input statistics; collision_updates (default 0) at collision_lr
            n_up = int(init["collision_updates"])
            c_lr, c_warm, count = float(init["collision_lr"]), int(init["collision_warmup"]), min(5, n_up) if smoke else n_up
        params = [p for p in coll.parameters() if p.requires_grad]
        if geo_cfg:
            params = params + list(model.geo_collision.parameters())
        opt = torch.optim.AdamW(params, lr=c_lr, weight_decay=0.0)
        ema2 = {}
        for update in range(1, count + 1):
            for g in opt.param_groups:
                g["lr"] = c_lr * cosine(update, count, c_warm, 0.01)
            loss = collision_loss(collision_batch(spec["batch"]), True)
            if geo_cfg:   # both terms divided by their running means (as in stage 3)
                gl = geo_term("collision", sample(geo_windows, 64 if smoke else spec["batch"] // 4))
                for name_, v_ in (("bce", loss), ("geo", gl)):
                    ema2[name_] = float(v_) if name_ not in ema2 else 0.99 * ema2[name_] + 0.01 * float(v_)
                loss = loss / max(ema2["bce"], 1e-30) + geo_weight * min(1.0, update / 1000) * gl / max(ema2["geo"], 1e-30)
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
        event("collision_fit", **report, **({"updates": count, "init_from": True} if packet is not None else {}))
        save(output / "collision.pt", "collision", 0)
    for p in coll.parameters():
        p.requires_grad_(False)

    # ---- 3. contact network, co-trained with the core -----------------------------
    cont = model.contact
    hit_windows = torch.nonzero(train & labels.any(-1))   # contact batches; rollout starts lie 1..horizon steps before
    # Optional contact sampling (config "contact_sampling": "sqrt_groups", absent = the released rule: uniform over
    # contact windows): group the contact windows by the set of pairs that are on and draw each group in proportion to
    # the square root of its size (as the core's bins), for the contact batches and the contact-near rollout starts.
    # Rare combinations (one jaw and the table) then get several times their natural share.
    # "sqrt_family_groups" groups by family and pair set (data with families): a rare family's holds (weak closure) are
    # then not drowned by the holds of normal grasps.
    sampling = cfg.get("contact_sampling")
    if sampling not in (None, "sqrt_groups", "sqrt_family_groups") or (sampling == "sqrt_family_groups" and family is None):
        raise ValueError(f"contact_sampling takes sqrt_groups or sqrt_family_groups (data with families), not {sampling}")
    if sampling:
        code_ = (labels[hit_windows[:, 0], hit_windows[:, 1]].long() * (2 ** torch.arange(n_pairs, device=dev))).sum(-1)
        if sampling == "sqrt_family_groups":
            code_ = code_ + (2 ** n_pairs) * family[hit_windows[:, 0]]
        hit_groups = [hit_windows[code_ == c] for c in torch.unique(code_).tolist()]
        hit_weights = torch.tensor([math.sqrt(len(g)) for g in hit_groups], dtype=torch.float64)
        hit_weights = (hit_weights / hit_weights.sum()).tolist()
        event("contact_sampling", rule=sampling, groups={str(int(torch.unique(code_)[k])): [len(g), round(w, 4)]
                                                          for k, (g, w) in enumerate(zip(hit_groups, hit_weights))})

    def sample_hits(n):
        if not sampling:
            return sample(hit_windows, n)
        return torch.cat([sample(g, max(1, int(round(n * w)))) for g, w in zip(hit_groups, hit_weights)])

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
    # Optional own-history guard rule (config "push_guard", absent in train.json = the released rule): "unit":
    # "all_hit" measures a deviation in RMS one-step changes over all contact windows of every body (the released
    # unit uses only single-contact steps that touch the body, which can be ~0 for a joint that does not move while
    # it touches, so that almost every own history fails); "bodies": "affected" judges only the bodies that a pair
    # of the window touches (an untouched body's own history is the core's, which the contact label does not judge).
    guard = cfg.get("push_guard") or {}
    if set(guard) - {"unit", "bodies"} or guard.get("unit", "single") not in ("single", "all_hit") \
            or guard.get("bodies", "all") not in ("all", "affected"):
        raise ValueError(f"push_guard takes unit: single | all_hit and bodies: all | affected, not {guard}")
    # Optional loss-scale rule (config "loss_scale", absent in train.json = the released rule: kappa(type) x the RMS of
    # each channel's contact residual on single-contact steps). "step_type": kappa(type) x the RMS of the one-step
    # change over all contact windows, pooled over the channels of one type in one body (all joint rates of an arm
    # share one scale; vx, vy, vz of a body share one). It sets the contact loss and, x rollout_scale_factor, the
    # rollout loss. The released rule gives a channel whose contact effect is ~0 (a joint that does not move while
    # it touches) a ~0 scale, so that channel dominates both losses.
    loss_rule = (cfg.get("loss_scale") or {}).get("rule")
    # "kappa": {channel type: value} replaces TYPE_RULES' kappa of those types in the "step_type" rule (e.g. the angular
    # velocity kappa equal to the velocity kappa, so that a cube's spin error counts like its speed error).
    kappa_over = (cfg.get("loss_scale") or {}).get("kappa") or {}
    contact_huber = cfg.get("contact_huber")   # absent = the released squared contact loss
    contact_huber = None if contact_huber is None else float(contact_huber)
    if cfg.get("loss_scale") and (set(cfg["loss_scale"]) - {"rule", "kappa"} or loss_rule != "step_type"
                                  or set(kappa_over) - set(TYPE_RULES)):
        raise ValueError(f"loss_scale takes rule: step_type and kappa {{type: value}}, not {cfg['loss_scale']}")

    def affected_of(g):
        """[n, D] 1 for each body that a pair with switch g [n, P] on touches."""
        affected = torch.zeros(len(g), n_mov, dtype=torch.float64, device=dev)
        for p in range(n_pairs):
            on = g[:, p] > 0.5
            affected[on, model.pair_pos[p, 0]] = 1.0
            if model.pair_mm[p]:
                affected[on, model.pair_pos[p, 1]] = 1.0
        return affected

    def contact_example(windows, train_mode, push=0.0, core_grad=False):
        """One step with the recorded switches; with probability `push` the history is the model's own (kept only
        if its last state lies within push_max_dev one-step RMS changes of the record)."""
        h, ok = hist_of(windows)
        g = labels[windows[:, 0], windows[:, 1]].to(torch.float64)
        affected = affected_of(g)
        if push > 0:
            sel = torch.nonzero(torch.rand(len(windows), device=dev, generator=rng) < push).flatten()
            if len(sel):
                m = torch.randint(1, cfg["push_max_steps"] + 1, (len(sel),), device=dev, generator=rng)
                hs, oks = own_history(windows[sel], m)
                if guard:
                    dev_b = ((hs[:, -1] - h[sel, -1]) * masks / guard_unit).abs().amax(-1)          # [n, D]
                    dev_ = (dev_b * affected[sel] if guard.get("bodies") == "affected" else dev_b).amax(-1)
                else:
                    dev_ = ((hs[:, -1] - h[sel, -1]) * masks / step_rms).abs().amax((-1, -2))
                keep = dev_ <= cfg["push_max_dev"]
                push_stats["tried"] += len(sel)
                push_stats["kept"] += int(keep.sum())
                sel, hs, oks = sel[keep], hs[keep], oks[keep]
                h, ok = h.clone(), ok.clone()
                h[sel], ok[sel] = hs, oks
        if train_mode:
            ok = drop_history(ok)
            h = kick_rates(shift_poses(h, "contact"), ok)
        pred = model.step_with_gates(h, ok, act_of(windows), g, sigma if train_mode else None, core_grad=core_grad)
        y = states[windows[:, 0], windows[:, 1] + stride]
        return pred, y, affected, h[:, -1]

    resume_path = output / "contact_state.pt"
    if (output / "contact_done.json").exists():
        best["score"] = json.loads((output / "contact_done.json").read_text())["best"]
        event("contact_loaded", best=best["score"])
    elif resume_path.exists():
        event("contact_resume_found")
    elif packet is not None:   # fine-tuning: the checkpoint's contact network maps and its run's loss scales
        (output / "contact_scale.json").write_text((Path(init["run"]) / "contact_scale.json").read_text())
        event("init_contact_scale", source=str(Path(init["run"]) / "contact_scale.json"),
              note="no input statistics, no linear fit, loss scales and step_rms of the source run")
    else:
        with torch.no_grad():
            cpools = []
            for p in range(n_pairs):
                w = torch.nonzero(labels[..., p] & train)
                cpools.append(w if len(w) > 200 else pools[p])
            fit_inputs(cont, cpools, "contact_inputs")
            # Output scale and linear path per pair from steps with exactly one contact (attributed to that pair).
            gt = labels[hit_windows[:, 0], hit_windows[:, 1]]
            single = hit_windows[gt.sum(-1) == 1]
            single = single[torch.randperm(len(single), device=dev, generator=rng)[:400000]]
            trace("contact_fit", single[:, 0])
            sp = labels[single[:, 0], single[:, 1]].float().argmax(-1)
            h, ok = hist_of(single)
            act = act_of(single)
            cur, diff, lag_ok = model.code(h, ok)
            core_d = torch.cat([model.run_core(*t) for t in zip(cur.split(65536), diff.split(65536), act.split(65536), lag_ok.split(65536))])
            resid = (states[single[:, 0], single[:, 1] + stride] - cur - core_d) * masks           # [n, D, S]
            fit_resid = resid.clone()   # integrated pose: the contact network's pose output is not used
            fit_resid[:, derived_cols] = 0.0   # nor its derived outputs
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
            extra_scale = {}
            if guard.get("unit") == "all_hit" or loss_rule:   # every body over a sample of all contact windows
                aw = hit_windows[torch.randperm(len(hit_windows), device=dev, generator=rng)[:400000]]
                trace("push_guard_unit", aw[:, 0])
                step_a = (states[aw[:, 0], aw[:, 1] + stride] - states[aw[:, 0], aw[:, 1]]) * masks
                r = step_a.square().mean(0).sqrt()
                extra_scale["step_rms_all"] = torch.where(masks > 0, r.clamp_min(1e-12), torch.ones_like(r)).tolist()
                if loss_rule:   # loss_scale "step_type": kappa(type) x RMS step change pooled over the body's channels of that type
                    pooled = torch.ones(n_mov, smax, dtype=torch.float64, device=dev)
                    for b, ts in enumerate(types):
                        for t in set(ts):
                            cols = torch.tensor([c for c, tt in enumerate(ts) if tt == t and masks[b, c] > 0], dtype=torch.long, device=dev)
                            if len(cols):
                                pooled[b, cols] = r[b, cols].square().mean().sqrt().clamp_min(1e-12)
                    extra_scale["scale_released_rule"] = sc.tolist()
                    kap = kappa.clone()
                    for d_, ts in enumerate(types):
                        for c_, t in enumerate(ts):
                            if t in kappa_over:
                                kap[d_, c_] = float(kappa_over[t])
                    sc = torch.where(masks > 0, kap * pooled, torch.ones_like(pooled))
        event("contact_linear_fit", loss_scale=sc.tolist(), step_rms=srms.tolist(), **extra_scale, **fit_report)
        (output / "contact_scale.json").write_text(json.dumps({"scale": sc.tolist(), "step_rms": srms.tolist(), **extra_scale}))

    _sc = json.loads((output / "contact_scale.json").read_text())
    scale = torch.tensor(_sc["scale"], dtype=torch.float64, device=dev)
    step_rms = torch.tensor(_sc["step_rms"], dtype=torch.float64, device=dev)
    guard_unit = torch.tensor(_sc["step_rms_all"], dtype=torch.float64, device=dev) if guard.get("unit") == "all_hit" else step_rms
    vpos = torch.nonzero(vwin & labels.any(-1))
    vpos = vpos[torch.randperm(len(vpos), device=dev, generator=rng)[: (200 if smoke else 20000)]]

    def onestep():
        """One-step contact error on validation contact steps (recorded and own histories), in loss scales."""
        trace("onestep", vpos[:, 0])
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
        trace("contact", windows[:, 0])
        pred, y, aff, _ = contact_example(windows, train_mode, push, core_grad)
        if contact_huber is None:
            err = ((pred - y) * core_masks / scale).square().sum(-1) / core_masks.sum(-1).clamp_min(1)
        else:   # config "contact_huber": Huber of the scaled error (delta in scale units), as the rollout loss
            z = (pred - y) * core_masks / scale
            err = F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=contact_huber).sum(-1) / core_masks.sum(-1).clamp_min(1)
        return (err * aff).sum() / aff.sum().clamp_min(1)

    rscale = scale * cfg["rollout_scale_factor"]

    # Optional relative loss (config "relative_loss": {"fk": KinematicChain spec path, "body": rigid body, "frame":
    # articulated body, "pairs": [pair names], "weight": w}; absent = none). A training signal only: the model's state
    # and inputs stay as they are. In the rollout loss, on the steps where all the given pairs are on in the record
    # (a held cube), the body's position and slip velocity in the frame of the frame body's point (forward kinematics
    # of the predicted and of the recorded joint state) are compared too, Huber in the rollout scales of the body's
    # position and velocity channels, mean over those steps, times w.
    rel_cfg = cfg.get("relative_loss")
    if rel_cfg:
        rel_kin = KinematicChain(json.loads((REPO_ROOT / rel_cfg["fk"]).read_text())).to(dev)
        names_ = [b["name"] for b in moving_b]
        rel_b, rel_f = names_.index(rel_cfg["body"]), names_.index(rel_cfg["frame"])
        ft, bt = types[rel_f], types[rel_b]
        rel_q = [c for c, t in enumerate(ft) if t == "angle"][: rel_kin.n]
        rel_qd = [c for c, t in enumerate(ft) if t == "angular_rate"][: rel_kin.n]
        rel_p = [c for c, t in enumerate(bt) if t == "position"]
        rel_o = [c for c, t in enumerate(bt) if t == "quaternion"]
        rel_v = [c for c, t in enumerate(bt) if t == "velocity"]
        pnames = [m.get("name") for m in system.get("pair_metadata", [])]
        rel_pairs = torch.tensor([pnames.index(n) for n in rel_cfg["pairs"]], device=dev)
        rel_w = float(rel_cfg.get("weight", 1.0))

    def rel_of(x):
        """[n, D, S] -> [n, 6]: position and slip velocity of the relative-loss body in the frame point's frame."""
        r = relative_channels(rel_kin(x[:, rel_f, rel_q], x[:, rel_f, rel_qd], model.dt), x[:, rel_b, rel_p],
                              x[:, rel_b, rel_o], x[:, rel_b, rel_v])
        return torch.cat((r[:, 0:3], r[:, 9:12]), -1)

    def rollout_loss(horizon, batch):
        """Huber loss of free rollouts (the model's own states, recorded commands) against the recorded trajectory;
        half of the starts lie 1..horizon steps before a contact step."""
        starts = sample_hits(batch // 2).clone()
        starts[:, 1] = (starts[:, 1] - torch.randint(1, max(2, horizon), (len(starts),), device=dev, generator=rng) * stride).clamp_min(0)
        w = torch.cat((starts, sample(any_train, max(0, batch - len(starts)))))
        trace("rollout", w[:, 0])
        h, ok = hist_of(w)
        current = h[:, -1]
        total, den = current.sum() * 0.0, 0
        rel_total, rel_den = current.sum() * 0.0, 0
        if rel_cfg:
            rel_scale = torch.cat((rscale[rel_b, rel_p], rscale[rel_b, rel_v]))
        for step in range(horizon):
            t_now = w[:, 1] + step * stride
            j = t_now + stride
            active = j < lengths[w[:, 0]]
            y = states[w[:, 0], j.clamp_max(n_rec - 1)]
            pred = model(h, ok, actions[w[:, 0], t_now.clamp_max(n_rec - 1)])
            err = F.huber_loss((pred - y) * masks / rscale, torch.zeros_like(pred), reduction="none", delta=1.0)
            total = total + (err.mean((-1, -2)) * active).sum()
            den += int(active.sum())
            if rel_cfg:
                held = active & labels[w[:, 0], t_now.clamp_max(n_win - 1)][:, rel_pairs].all(-1)
                if held.any():
                    e = (rel_of(pred[held]) - rel_of(y[held])) / rel_scale
                    rel_total = rel_total + F.huber_loss(e, torch.zeros_like(e), reduction="none", delta=1.0).mean(-1).sum()
                    rel_den += int(held.sum())
            current = torch.where(active[:, None, None], pred, current)
            h, ok = model.advance(h, ok, current)
        out = total / max(1, den)
        return out + rel_w * rel_total / max(1, rel_den) if rel_cfg else out

    # Optional co-training of the collision network in this stage (config "collision_cotrain", absent in train.json =
    # the released rule: frozen after stage 2): its binary cross-entropy on the stage-2 batches (contact, near-contact
    # and any steps) joins the joint loss, with weight "weight" (default 1, divided by its running mean like the other
    # terms), batch "batch" (default the stage-2 batch) and learning-rate factor "lr_scale" (default 1). With "push"
    # (default true) a fraction push_prob of its histories are the model's own 1-3 steps after push_start, kept if
    # every body lies within push_max_dev guard units of the record (the switch then sees the states it meets in a
    # rollout, with the recorded label).
    coll_on = cfg.get("collision_cotrain") is not None   # {} = on, with the defaults
    coll_co = cfg.get("collision_cotrain") or {}
    if set(coll_co) - {"weight", "batch", "lr_scale", "push"}:
        raise ValueError(f"collision_cotrain takes weight, batch, lr_scale and push, not {coll_co}")
    coll_push = {"tried": 0, "kept": 0}

    def collision_co_loss(n, push):
        picks = collision_batch(n)
        w = picks[:, :2]
        h, ok = hist_of(w)
        if push > 0:
            sel = torch.nonzero(torch.rand(len(w), device=dev, generator=rng) < push).flatten()
            if len(sel):
                m = torch.randint(1, cfg["push_max_steps"] + 1, (len(sel),), device=dev, generator=rng)
                hs, oks = own_history(w[sel], m)
                keep = ((hs[:, -1] - h[sel, -1]) * masks / guard_unit).abs().amax((-1, -2)) <= cfg["push_max_dev"]
                coll_push["tried"] += len(sel)
                coll_push["kept"] += int(keep.sum())
                sel, hs, oks = sel[keep], hs[keep], oks[keep]
                h, ok = h.clone(), ok.clone()
                h[sel], ok[sel] = hs, oks
        ok = drop_history(ok)
        h = shift_poses(h, "collision")
        cur, diff, lag_ok = model.code(h, ok, sigma)
        logits = model.logits(cur, diff, act_of(w), lag_ok, torch.arange(len(picks), device=dev), picks[:, 2])
        return F.binary_cross_entropy_with_logits(logits, labels[picks[:, 0], picks[:, 1], picks[:, 2]].to(torch.float64))

    if not (output / "contact_done.json").exists():
        # L_core + L_contact + L_trajectory, each divided by its running mean
        for p in core.parameters():
            p.requires_grad_(True)
        joint = [p for p in list(core.parameters()) + list(cont.parameters()) if p.requires_grad]
        if geo_cfg:
            joint = joint + list(model.geo_contact.parameters())
        count = 5 if smoke else co["updates"]
        joint_coll = []
        if coll_on:
            for name_, p in coll.named_parameters():
                p.requires_grad_(name_ != "lin")   # the collision network has no linear path
            joint_coll = [p for p in coll.parameters() if p.requires_grad]
            if geo_cfg:
                joint_coll = joint_coll + list(model.geo_collision.parameters())
            opt = torch.optim.AdamW([{"params": joint, "lr_scale": 1.0},
                                     {"params": joint_coll, "lr_scale": float(coll_co.get("lr_scale", 1.0))}],
                                    lr=co["lr"], weight_decay=0.0)
        else:
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
        elif packet is not None and init["check_first"]:   # the starting model is the first candidate for best.pt
            check("init", 0, 0.0, {**onestep(), "note": "init_from checkpoint before the first update"})
            push_stats.update(tried=0, kept=0)
        h0 = max(1, round(co["horizon_start_s"] / model.dt))
        h1 = max(h0, round(co["horizon_max_s"] / model.dt))
        push_from = int(cfg["push_start"] * count)
        traj_every = co["traj_every"]
        for update in range(first, count + 1):
            frac = update / count
            for g in opt.param_groups:
                g["lr"] = co["lr"] * cosine(update, count, co["warmup"], 0.01) * g.get("lr_scale", 1.0)
            hb, okb, ab, bb, tb = core_batch(64 if smoke else co["core_batch"])
            okb = drop_history(okb)
            kind = model.body_kind[bb]
            kscale = torch.stack([core.out_std[k, k * smax:(k + 1) * smax] for k in range(nk)])[kind]
            terms = {"core": (((core_pred(hb, okb, ab, bb, sigma) - tb) / kscale).square() * core_masks[bb]).sum() / core_masks[bb].sum()}
            push = cfg["push_prob"] if update > push_from else 0.0
            terms["contact"] = contact_loss(sample_hits(64 if smoke else co["contact_batch"]), push, core_grad=True)
            if geo_cfg:
                gw_ = sample(geo_windows, 64 if smoke else co["contact_batch"] // 4)
                terms["geo"] = geo_term("contact", gw_) + (geo_term("collision", gw_) if coll_on else 0.0)
            if update % traj_every == 0:
                horizon = 4 if smoke else round(h0 + (h1 - h0) * min(1.0, frac / co["horizon_ramp"]))
                terms["traj"] = rollout_loss(horizon, 4 if smoke else co["traj_batch"])
            if coll_on:
                terms["collision"] = collision_co_loss(64 if smoke else int(coll_co.get("batch", cfg["collision"]["batch"])),
                                                       push if coll_co.get("push", True) else 0.0)
            total = 0.0
            for name, value in terms.items():
                v = float(value.detach())
                ema[name] = v if name not in ema else 0.99 * ema[name] + 0.01 * v
                weight = traj_every * min(1.0, frac / co["traj_ramp"]) if name == "traj" else \
                    float(coll_co.get("weight", 1.0)) if name == "collision" else \
                    geo_weight * min(1.0, update / 1000) if name == "geo" else 1.0
                total = total + weight * value / max(ema[name], 1e-30)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(joint, 1.0)
            if joint_coll:   # the collision network is clipped on its own
                gnorm = gnorm + torch.nn.utils.clip_grad_norm_(joint_coll, 1.0)
            if not torch.isfinite(gnorm):
                event("cotrain_nonfinite_gradient", update=update)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            if update % (5 if smoke else cfg["evaluate_every"]) == 0 or update == count:
                sigma = core_sigma(core_val())   # the noise follows the current core's one-step error
                # push_kept_fraction (as released): the training pushes since the last check and those of onestep()
                extra = {"collision_push_kept_fraction": coll_push["kept"] / max(1, coll_push["tried"])} if coll_on else {}
                check("cotrain", update, total.detach(), {**onestep(), **{f"ema_{k}": v for k, v in ema.items()},
                                                          "push_kept_fraction": push_stats["kept"] / max(1, push_stats["tried"]),
                                                          **extra})
                push_stats.update(tried=0, kept=0)
                coll_push.update(tried=0, kept=0)
                tmp = output / "contact_state.pt.tmp"
                torch.save(dict(model_state_dict=model.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                                best=best["score"], update=update, ema=ema, sigma=sigma.tolist()), tmp)
                tmp.replace(resume_path)
        (output / "contact_done.json").write_text(json.dumps({"best": best["score"]}))

    # ---- 4. final validation of best.pt -------------------------------------------
    if not (output / "complete.json").exists():
        model_best, metadata = load_model(output / "best.pt", dev)
        trace("final_validation", val)
        with torch.no_grad():
            report = file_rows(evaluate(model_best, data, val, by_family=True), data)
        atomic_json(output / "validation.json", report)
        chosen = report.get("selected", report)   # the families that checkpoint selection used
        done = dict(complete=True, smoke=smoke, test_used_for_selection=False, total_elapsed_s=time.perf_counter() - start,
                    checkpoint_sha256=file_sha256(output / "best.pt"), best_stage=metadata["stage"],
                    best_update=metadata["update"], validation_score=chosen["selection_score"], checkpoints_compared=checks["count"])
        if packet is not None:
            done.update(init_from=init["run"], init_checkpoint=init["checkpoint"])
        if "evaluation" in system:
            done.update(score_terms={t["stat"]: t["value"] for t in chosen["score_terms"]}, finite=report["finite_fraction"],
                        family_scores={k: v["selection_score"] for k, v in report.get("families", {}).items()})
        else:
            done.update(target_pos_median_mm=1000 * report["target_pos_m"]["median"], target_pos_p95_mm=1000 * report["target_pos_m"]["p95"],
                        target_angle_median_deg=math.degrees(report["target_angle_rad"]["median"]),
                        target_angle_p95_deg=math.degrees(report["target_angle_rad"]["p95"]),
                        moved_pos_median_mm=1000 * report["moved_target_pos_m"]["median"],
                        moved_angle_median_deg=math.degrees(report["moved_target_angle_rad"]["median"]),
                        path_p95_mm=1000 * report["path_rmse_m"]["p95"], ee_p95_mm=1000 * report["ee_rmse_m"]["p95"],
                        events_ok=report["event_ok_fraction"], finite=report["finite_fraction"])
        atomic_json(output / "complete.json", done)
        event("complete", **done)
    if drawn is not None:
        atomic_json(args.trace_draws, {k: sorted(v) for k, v in drawn.items()})


if __name__ == "__main__":
    main()
