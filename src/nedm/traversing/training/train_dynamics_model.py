"""Train the NRD vehicle-dynamics model on recorded rigid and soil drives on f104 (the tracker's training world).

Data: the per-episode cache (``dynamics_data``); train = the manifest's train groups, val = its val groups (model
selection and gates), test never loaded. Every batch takes ``--domain-frac`` of its windows from soil, the rest
from rigid ground. Loss per step = teacher-forced one-step loss over the 16-frame context + ``--rollout-weight`` x an
``--rollout-steps``-step fed-back rollout loss (state fed back, crop re-taken at the dead-reckoned pose, recorded
actions); Huber on the normalised z1 delta (channel weights from ``--delta-scale``) and on the normalised power;
transitions whose action was not held (cache ``hold_ok`` False) weigh ``--hold-bad-weight``. AdamW, linear warm-up
then cosine decay to ``--min-lr``, gradient clip 1.

Validation every ``--eval-every`` steps on the val groups, per ground type, over ``--val-steps`` (60) frames: random
windows, stalled windows (every predicted frame stalled), moving windows (no stalled frame, mean |vx| > 0.5 m/s) and
brake-onset windows, teacher-forced and fed back. ``ckpt_best.pt`` = lowest mean over ground types of the fed-back
random-window normalised z1 error at 60 frames. Gates (fed back, per ground type): stalled windows escape (> 1 m)
less than 20 %, moving-window displacement error below 25 %, brake-onset speed-response error below 25 %.

    python traversing/training/train_dynamics_model.py --out runs/nrd_tag_v3 --cond tag --crop-k 8 --crop-half-m 6 \\
        --steps 30000 --batch 256 --rollout-steps 8 --delta-scale --eval-every 2000 --val-max-per-domain 256 \\
        --ckpt-every-min 15 --max-minutes 225
    # continue an interrupted run with the same arguments plus --resume runs/nrd_tag_v3/ckpt_last.pt
    # smoke: --max-episodes 40 --steps 200 --eval-every 100 --n-layer 2 --n-embd 64

Outputs in ``--out``: config.json, train_log.jsonl, metrics.json, ckpt_best.pt, ckpt_last.pt, run_state.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dynamics_data as D
from dynamics_model import integrate_pose, load_grid, model_config, NRDModel, save_nrd
from state import VX, YAW_RATE

GATE = {"stalled_escape_frac": 0.20, "moving_disp_rel_err": 0.25, "brake_vx_resp_rel_err": 0.25}
# arguments that change the data, the model, the loss or the optimiser: --resume refuses any difference (the schedule
# --lr/--min-lr/--warmup-steps/--steps is checked separately, the grid and the cache manifest by sha256)
RESUME_FIXED = ("cache", "cond", "crop_k", "crop_half_m", "domain_frac", "max_frames", "max_episodes", "context", "batch",
                "n_layer", "n_head", "n_embd", "token_dim", "rollout_steps", "rollout_weight", "delta_scale",
                "hold_bad_weight", "weight_decay", "grad_clip", "seed")


# ------------------------------------------------------------------------------------------ losses
def _wmean(per: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    return (per * wt).sum() / wt.sum().clamp_min(1.0)


def step_loss(model, batch, context: int, w, w_bad: float):
    """Teacher-forced one-step deltas over frames t0 .. t0+context, tokens at the recorded poses."""
    z1, act, pose = batch["z1"][:, :context + 1], batch["act"][:, :context + 1], batch["pose"][:, :context + 1]
    token = model.token(pose)
    z_in = z1[:, :-1]
    delta, power = model(z_in, token[:, :-1], act[:, :-1], batch["domain"])
    tgt = z1[:, 1:] - z_in
    wt = torch.where(batch["hold"][:, :context] > 0.5, torch.ones_like(delta[..., 0]), torch.full_like(delta[..., 0], w_bad))
    res = (delta - tgt) * w if w is not None else (delta - tgt)
    per = F.huber_loss(res, torch.zeros_like(res), delta=1.0, reduction="none").mean(-1)
    l_z1 = _wmean(per, wt)
    l_p = _wmean(F.huber_loss(power, batch["power"][:, 1:context + 1], delta=1.0, reduction="none").mean(-1), wt)
    return l_z1 + l_p, {"z1": float(l_z1.detach()), "power": float(l_p.detach())}


def rollout_loss(model, batch, context: int, steps: int, z1_mean, z1_std, w, w_bad: float):
    """``steps``-step autoregressive loss under the recorded actions: state fed back, crop re-taken at the
    dead-reckoned pose every step, each step weighted by the hold flag of its recorded transition."""
    z1, act, pose_gt, dom = batch["z1"], batch["act"], batch["pose"], batch["domain"]
    token_hist = model.token(pose_gt[:, :context])
    z1_hist = z1[:, :context]
    pose = pose_gt[:, context - 1]
    l_z1 = l_p = 0.0
    for step in range(steps):
        window = slice(step, step + context)
        delta, power = model(z1_hist[:, -context:], token_hist[:, -context:], act[:, window], dom)
        z1_next = z1_hist[:, -1] + delta[:, -1]
        tgt = z1[:, context + step]
        wt = torch.where(batch["hold"][:, context - 1 + step] > 0.5, torch.ones_like(tgt[:, 0]), torch.full_like(tgt[:, 0], w_bad))
        res = (z1_next - tgt) * w if w is not None else (z1_next - tgt)
        l_z1 = l_z1 + _wmean(F.huber_loss(res, torch.zeros_like(res), delta=1.0, reduction="none").mean(-1), wt)
        l_p = l_p + _wmean(F.huber_loss(power[:, -1], batch["power"][:, context + step], delta=1.0, reduction="none").mean(-1), wt)
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        nxt = model.token(pose.unsqueeze(1))[:, 0]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        token_hist = torch.cat([token_hist, nxt.unsqueeze(1)], dim=1)
    parts = {"ro_z1": float(l_z1.detach()) / steps, "ro_power": float(l_p.detach()) / steps}
    return (l_z1 + l_p) / steps, parts


# ------------------------------------------------------------------------------------------ validation
def find_windows(b: D.Batcher, norm: D.Normalizer, kind: str, context: int, K: int, rng, max_per_domain: int,
                 stride: int = 10) -> dict:
    """domain -> (ep, t0) anchors of one window kind; the rollout predicts frames t0+context .. t0+context+K-1 from
    the state at frame t0+context-1 under the recorded actions act[t0+context-1 ..]."""
    L = context + K
    out = {}
    for d, eps in b.dom_eps.items():
        cand = []
        for e in eps:
            n = int(b.n_valid[e])
            if n < L:
                continue
            if kind == "random":
                cand.extend((e, t) for t in range(0, n - L + 1))
            elif kind == "stalled":  # every predicted frame inside one run of stalled frames [t, u)
                st = b.stalled[e, :n]
                t = 0
                while t < n:
                    if st[t]:
                        u = t
                        while u < n and st[u]:
                            u += 1
                        for t0 in range(max(t - context, 0), u - K - context + 1, stride):
                            cand.append((e, t0))
                        t = u
                    else:
                        t += 1
            elif kind == "moving":
                st = b.stalled[e, :n]
                for t0 in range(0, n - L + 1, stride):
                    if not st[t0:t0 + L].any():
                        cand.append((e, t0))
            elif kind == "brake":  # first frame with brake > 0.2 after >= 10 frames below
                brk = b.act[e, :n, 2] * float(norm.act_std[2]) + float(norm.act_mean[2])
                for f in range(max(10, context - 1), n - K - 1):
                    if brk[f] > 0.2 and (brk[f - 10:f] < 0.2).all():
                        t0 = f - context + 1
                        if t0 >= 0 and t0 + L <= n:
                            cand.append((e, t0))
            else:
                raise ValueError(kind)
        if kind == "moving" and cand:  # keep windows where the vehicle actually moves
            keep = []
            for e, t0 in np.asarray(cand):
                vx = b.z1[e, t0 + context:t0 + context + K, VX] * float(norm.z1_std[VX]) + float(norm.z1_mean[VX])
                if np.abs(vx).mean() > 0.5:
                    keep.append((e, t0))
            cand = keep
        if not cand:
            continue
        arr = np.asarray(cand, np.int64)
        if len(arr) > max_per_domain:
            arr = arr[rng.choice(len(arr), max_per_domain, replace=False)]
        out[d] = (arr[:, 0], arr[:, 1])
    return out


@torch.no_grad()
def rollout_windows(model, b: D.Batcher, norm, ep: np.ndarray, t0: np.ndarray, context: int, K: int, mode: str, device,
                    chunk: int = 256) -> dict:
    """Per window: along-track displacement (pred / rec, m), vx and yaw-rate change over the window and trajectory
    MAE, normalised z1 MAE and pose error at K. ``mode`` 'fedback': state and pose fed back, crop at the dead-reckoned
    pose; 'teacher': true state history and poses every step, the pose integrated from the one-step predictions."""
    model.eval()
    z1_mean = torch.tensor(norm.z1_mean, dtype=torch.float32, device=device)
    z1_std = torch.tensor(norm.z1_std, dtype=torch.float32, device=device)
    res = {k: [] for k in ("disp_pred", "disp_rec", "dvx_pred", "dvx_rec", "dyr_pred", "dyr_rec", "vx_mae", "yr_mae",
                           "vx_rec_abs", "yr_rec_abs", "z1_mae_norm", "pose_err_m", "yaw_err_deg")}
    L = context + K
    for i in range(0, len(ep), chunk):
        bt = b.window_batch(ep[i:i + chunk], t0[i:i + chunk], L, device)
        z1, act, pose_gt, dom = bt["z1"], bt["act"], bt["pose"], bt["domain"]
        z1_hist = z1[:, :context]
        token_all = model.token(pose_gt) if mode == "teacher" else None
        token_hist = token_all[:, :context] if mode == "teacher" else model.token(pose_gt[:, :context])
        pose = pose_gt[:, context - 1].clone()
        pose0 = pose_gt[:, context - 1]
        preds = []
        for step in range(K):
            window = slice(step, step + context)
            if mode == "teacher":
                delta, _ = model(z1[:, window], token_all[:, window], act[:, window], dom)
                z1_next = z1[:, context - 1 + step] + delta[:, -1]
            else:
                delta, _ = model(z1_hist[:, -context:], token_hist[:, -context:], act[:, window], dom)
                z1_next = z1_hist[:, -1] + delta[:, -1]
                z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
            pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
            if mode != "teacher":
                token_hist = torch.cat([token_hist, model.token(pose.unsqueeze(1))[:, 0].unsqueeze(1)], dim=1)
            preds.append(z1_next)
        pred = torch.stack(preds, 1)  # (B, K, 17) normalised
        rec = z1[:, context:context + K]
        pred_phys, rec_phys = pred * z1_std + z1_mean, rec * z1_std + z1_mean
        start_phys = z1[:, context - 1] * z1_std + z1_mean
        gt_end = pose_gt[:, context + K - 1]
        c0, s0 = torch.cos(pose0[:, 2]), torch.sin(pose0[:, 2])
        disp = lambda p: c0 * (p[:, 0] - pose0[:, 0]) + s0 * (p[:, 1] - pose0[:, 1])
        yaw_err = torch.rad2deg(torch.abs((pose[:, 2] - gt_end[:, 2] + math.pi) % (2 * math.pi) - math.pi))
        add = lambda k, v: res[k].append(v.detach().cpu().numpy())
        add("disp_pred", disp(pose)); add("disp_rec", disp(gt_end))
        add("dvx_pred", pred_phys[:, -1, VX] - start_phys[:, VX]); add("dvx_rec", rec_phys[:, -1, VX] - start_phys[:, VX])
        add("dyr_pred", pred_phys[:, -1, YAW_RATE] - start_phys[:, YAW_RATE]); add("dyr_rec", rec_phys[:, -1, YAW_RATE] - start_phys[:, YAW_RATE])
        add("vx_mae", (pred_phys[:, :, VX] - rec_phys[:, :, VX]).abs().mean(1)); add("yr_mae", (pred_phys[:, :, YAW_RATE] - rec_phys[:, :, YAW_RATE]).abs().mean(1))
        add("vx_rec_abs", rec_phys[:, :, VX].abs().mean(1)); add("yr_rec_abs", rec_phys[:, :, YAW_RATE].abs().mean(1))
        add("z1_mae_norm", (pred[:, -1] - rec[:, -1]).abs().mean(1)); add("pose_err_m", (pose[:, :2] - gt_end[:, :2]).norm(dim=1)); add("yaw_err_deg", yaw_err)
    model.train()
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in res.items()}


def cell_metrics(r: dict) -> dict:
    n = len(r["disp_pred"])
    if n == 0:
        return {"n": 0}
    f = lambda a: float(np.mean(a))
    rel = lambda num, den: float(np.mean(np.abs(num)) / max(np.mean(np.abs(den)), 1e-3))
    return {"n": int(n),
            "disp_signed_err_m": f(r["disp_pred"] - r["disp_rec"]), "disp_abs_err_m": f(np.abs(r["disp_pred"] - r["disp_rec"])),
            "disp_rel_err": rel(r["disp_pred"] - r["disp_rec"], r["disp_rec"]), "disp_rec_mean_m": f(r["disp_rec"]),
            "disp_pred_mean_m": f(r["disp_pred"]),
            "escape_frac": f(np.abs(r["disp_pred"]) > 1.0), "rec_escape_frac": f(np.abs(r["disp_rec"]) > 1.0),
            "vx_resp_rel_err": rel(r["dvx_pred"] - r["dvx_rec"], r["dvx_rec"]), "dvx_rec_mean_mps": f(r["dvx_rec"]),
            "dvx_pred_mean_mps": f(r["dvx_pred"]),
            "vx_mae_mps": f(r["vx_mae"]), "vx_mae_rel": rel(r["vx_mae"], r["vx_rec_abs"]),
            "yawrate_resp_rel_err": rel(r["dyr_pred"] - r["dyr_rec"], r["dyr_rec"]), "yawrate_mae_radps": f(r["yr_mae"]),
            "yawrate_mae_rel": rel(r["yr_mae"], r["yr_rec_abs"]),
            "z1_mae_norm": f(r["z1_mae_norm"]), "pose_err_m": f(r["pose_err_m"]), "yaw_err_deg": f(r["yaw_err_deg"])}


def evaluate(model, val: D.Batcher, norm, context: int, K: int, device, max_per_domain: int, seed: int) -> dict:
    """Flat metric dict: val/<domain>/<kind>/<mode>/<metric>, gate/<name>/<domain>, gate/pass/<domain>, sel."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for kind in ("random", "stalled", "moving", "brake"):
        wins = find_windows(val, norm, kind, context, K, rng, max_per_domain)
        raw = {"teacher": [], "fedback": []}
        for d, (ep, t0) in wins.items():
            for mode in ("teacher", "fedback"):
                r = rollout_windows(model, val, norm, ep, t0, context, K, mode, device)
                raw[mode].append(r)
                for k, v in cell_metrics(r).items():
                    out[f"val/{D.DOMAIN_VOCAB[d]}/{kind}/{mode}/{k}"] = v
        for mode in ("teacher", "fedback"):  # pooled over domains (window-weighted, no second rollout)
            pooled = raw[mode]
            if pooled:
                merged = {k: np.concatenate([p[k] for p in pooled]) for k in pooled[0]}
                for k, v in cell_metrics(merged).items():
                    out[f"val/all/{kind}/{mode}/{k}"] = v
    gate_src = {"stalled_escape_frac": ("stalled", "escape_frac"), "moving_disp_rel_err": ("moving", "disp_rel_err"),
                "brake_vx_resp_rel_err": ("brake", "vx_resp_rel_err")}
    for dom in list(D.DOMAIN_VOCAB) + ["all"]:
        flags = []
        for name, (kind, metric) in gate_src.items():
            key = f"val/{dom}/{kind}/fedback/{metric}"
            if key in out:
                out[f"gate/{name}/{dom}"] = out[key]
                out[f"gate/{name}/{dom}/n"] = out[f"val/{dom}/{kind}/fedback/n"]
                flags.append(bool(out[key] < GATE[name]))
        if flags:
            out[f"gate/pass/{dom}"] = bool(all(flags)) and len(flags) == len(GATE)  # a missing cell cannot pass
            out[f"gate/n_cells/{dom}"] = len(flags)
    sels = [out[f"val/{d}/random/fedback/z1_mae_norm"] for d in D.DOMAIN_VOCAB if f"val/{d}/random/fedback/z1_mae_norm" in out]
    out["sel"] = float(np.mean(sels)) if sels else float("nan")
    return out


def gate_report(m: dict) -> str:
    lines = []
    for dom in list(D.DOMAIN_VOCAB) + ["all"]:
        parts = [f"{name} {m[f'gate/{name}/{dom}']:.3f} (<{thr}, n={m[f'gate/{name}/{dom}/n']})"
                 for name, thr in GATE.items() if f"gate/{name}/{dom}" in m]
        if parts:
            lines.append(f"  gate[{dom}] pass={m.get(f'gate/pass/{dom}')}: " + "; ".join(parts))
    return "\n".join(lines) if lines else "  gate: no windows"


# ------------------------------------------------------------------------------------------ main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=D.DEFAULT_CACHE, help="per-episode cache directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default=D.DEFAULT_GRID, help="f104 terrain grid for the crops")
    ap.add_argument("--twin-split", default=D.DEFAULT_TWIN_SPLIT, help="twin group split to cross-check the cache split against ('none' skips)")
    ap.add_argument("--cond", choices=["tag"], default="tag", help="ground type one-hot on every token")
    ap.add_argument("--crop-k", type=int, default=8)
    ap.add_argument("--crop-half-m", type=float, default=6.0)
    ap.add_argument("--domain-frac", type=float, default=0.5, help="fraction of each batch from soil, in [0, 1]")
    ap.add_argument("--max-frames", type=int, default=1200, help="episodes are cut to this many frames")
    ap.add_argument("--max-episodes", type=int, default=0, help="subsample train AND val episodes (smoke runs)")
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--token-dim", type=int, default=64)
    ap.add_argument("--rollout-steps", type=int, default=8, help="K > 0 adds the K-step fed-back rollout loss")
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--delta-scale", action="store_true", help="equal-domain channel weights (dynamics_data.domain_delta_weights)")
    ap.add_argument("--hold-bad-weight", type=float, default=0.25, help="loss weight of transitions with hold_ok False")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--val-steps", type=int, default=60, help="rollout horizon of the validation windows (frames)")
    ap.add_argument("--val-max-per-domain", type=int, default=256, help="windows per kind and domain")
    ap.add_argument("--val-batches", type=int, default=20, help="batches of the validation one-step loss")
    ap.add_argument("--ckpt-every-min", type=float, default=15.0)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop (after saving ckpt_last) past this wall time since the start, data loading included")
    ap.add_argument("--stop-at-step", type=int, default=0, help="stop (after saving ckpt_last) at this step; the schedule keeps --steps")
    ap.add_argument("--resume", default="", help="ckpt_last.pt of an interrupted run (same data / model / loss / schedule args)")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args(argv)
    assert 0.0 <= args.domain_frac <= 1.0, "--domain-frac must lie in [0, 1]"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    t_main = time.time()  # --max-minutes counts from here

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = D.resolve_path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    man = D.read_manifest(args.cache)
    D.check_group_split_consistency(man)
    split_check = D.check_split_against_twin(man, args.twin_split)
    held = D.heldout_groups(man)
    train_keys = D.select_keys(man, splits=("train",), max_episodes=args.max_episodes, seed=args.seed)
    val_keys = D.select_keys(man, splits=("val",), max_episodes=args.max_episodes, seed=args.seed + 1)
    if not train_keys or not val_keys:
        raise SystemExit(f"empty split: train {len(train_keys)} val {len(val_keys)}")
    train_data = D.load_cache(args.cache, train_keys, args.max_frames)
    val_data = D.load_cache(args.cache, val_keys, args.max_frames)
    assert not (set(train_data.group) & held), "held-out groups in the training set"
    assert all(s == "train" for s in train_data.split) and all(s == "val" for s in val_data.split)
    n_test = sum(1 for k in man["episodes"] if man["split_of"][k] == "test")
    cnt = lambda d: {D.DOMAIN_VOCAB[i]: int((d.domain == i).sum()) for i in range(2)}
    print(f"cache {args.cache}: train {train_data.n_episodes} {cnt(train_data)} groups {len(set(train_data.group))} | "
          f"val {val_data.n_episodes} {cnt(val_data)} groups {len(set(val_data.group))} | test {n_test} episodes untouched | "
          f"frames <= {train_data.n_frames}; loaded in {time.time() - t_load:.1f}s", flush=True)
    print(f"  train recorded frames: mean {train_data.n_valid.mean():.0f} min {train_data.n_valid.min()} max {train_data.n_valid.max()}; "
          f"hold_ok {train_data.hold_ok[train_data.valid].mean():.3f}; stalled frames {int(train_data.stalled.sum())}", flush=True)

    grid_path = args.grid
    grid_sha = D.file_sha256(D.resolve_path(grid_path))
    man_sha = D.file_sha256(D.resolve_path(args.cache) / "cache_manifest.json")
    resume_payload = torch.load(D.resolve_path(args.resume), map_location="cpu", weights_only=False) if args.resume else None
    if resume_payload is not None:
        if "train_state" not in resume_payload:
            raise SystemExit("--resume needs a ckpt_last.pt with train_state")
        norm = D.Normalizer.from_dict(resume_payload["normalization"])
        cfg = dict(resume_payload["config"])
        sched = resume_payload["train_state"]["schedule"]
        mine = {"lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps, "steps": args.steps}
        if sched != mine:
            raise SystemExit(f"schedule mismatch on resume: checkpoint {sched} vs args {mine}")
        if resume_payload["grid_sha256"] != grid_sha:
            raise SystemExit("grid differs from the checkpoint's")
        prev_args = resume_payload.get("train_args")
        if not isinstance(prev_args, dict):
            raise SystemExit("--resume: the checkpoint carries no train_args to check the arguments against")
        diff = {k: {"checkpoint": prev_args.get(k), "args": vars(args)[k]} for k in RESUME_FIXED if prev_args.get(k) != vars(args)[k]}
        if diff:
            raise SystemExit(f"argument mismatch on resume (these change the data / model / loss / optimiser): {json.dumps(diff)}")
        if resume_payload.get("cache_manifest_sha256") != man_sha:
            raise SystemExit("cache manifest differs from the checkpoint's")
    else:
        norm = D.fit_normalizer(train_data)
        cfg = model_config(crop_k=args.crop_k, crop_half_m=args.crop_half_m, block_size=args.context,
                           n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, token_dim=args.token_dim)
    context = int(cfg["block_size"])
    train = D.Batcher(train_data, norm, context)
    val = D.Batcher(val_data, norm, context)
    del train_data, val_data

    loss_w = None
    if resume_payload is not None and resume_payload.get("delta_scale") is not None:
        loss_w = torch.tensor(resume_payload["delta_scale"], dtype=torch.float32, device=device)
    elif args.delta_scale:
        w = D.domain_delta_weights(train)
        loss_w = torch.tensor(w, dtype=torch.float32, device=device)
        print("channel weights:", np.round(w, 2).tolist(), flush=True)

    model = NRDModel(cfg, load_grid(str(D.resolve_path(grid_path))), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model crop {cfg['crop_k']}x{cfg['crop_k']} +-{cfg['crop_half_m']} m token {cfg['token_dim']} "
          f"L{cfg['n_layer']} H{cfg['n_head']} E{cfg['n_embd']} params {n_params / 1e6:.2f}M on {device}", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    schedule = {"lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps, "steps": args.steps}

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))

    rng = np.random.default_rng(args.seed)
    start_step, best, elapsed_before = 0, {"metric": float("inf"), "step": -1}, 0.0
    log = out_dir / "train_log.jsonl"
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
        ts = resume_payload["train_state"]
        opt.load_state_dict(ts["optimizer"])
        start_step = int(ts["step"])
        best = ts.get("best", best)
        elapsed_before = float(ts.get("elapsed_s", 0.0))
        rng.bit_generator.state = ts["np_rng"]
        torch.set_rng_state(ts["torch_rng"])
        if device == "cuda" and ts.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ts["cuda_rng"])
        print(f"resumed {args.resume} at step {start_step} (best {best.get('metric')} @ {best.get('step')}); "
              f"schedule {schedule}", flush=True)
        with log.open("a") as fh:
            fh.write(json.dumps({"phase": "resume", "step": start_step, "from": str(args.resume)}) + "\n")
        # the record of the run = the checkpoint's arguments; only the run-control arguments come from this command
        free = [k for k in vars(args) if k not in RESUME_FIXED and k not in ("lr", "min_lr", "warmup_steps", "steps")]
        cfg_args = {**resume_payload["train_args"], **{k: vars(args)[k] for k in free}, "resumed_from": str(args.resume)}
    else:
        log.write_text("")
        cfg_args = dict(vars(args))
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg_args, "model": cfg, "n_params": n_params, "device": device, "grid_sha256": grid_sha, "split_cross_check": split_check,
         "split_counts": {"train": train.n_episodes, "val": val.n_episodes, "test_untouched": n_test},
         "train_groups": len(set(man["group_of"][k] for k in train_keys)), "val_groups": len(set(man["group_of"][k] for k in val_keys)),
         "normalization": norm.to_dict(), "delta_scale": None if loss_w is None else loss_w.cpu().tolist(),
         "gate_thresholds": GATE}, indent=2))

    z1_mean_t = torch.tensor(norm.z1_mean, dtype=torch.float32, device=device)
    z1_std_t = torch.tensor(norm.z1_std, dtype=torch.float32, device=device)
    t_start = last_ckpt = time.time()
    last_metrics: dict = dict(resume_payload.get("metrics", {})) if resume_payload is not None else {}

    def train_state(step: int) -> dict:
        return {"optimizer": opt.state_dict(), "step": int(step), "schedule": schedule, "best": best,
                "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                "elapsed_s": elapsed_before + (time.time() - t_start)}

    def save(path: Path, step: int, m: dict, with_state: bool) -> None:
        save_nrd(path, model, norm, step, m, grid_path, grid_sha, loss_w.cpu().tolist() if loss_w is not None else None,
                 extra={"train_args": cfg_args, "cache": str(args.cache), "cache_manifest_sha256": man_sha},
                 train_state=train_state(step) if with_state else None)

    def write_metrics(step: int, m: dict) -> None:
        (out_dir / "metrics.json").write_text(json.dumps({"step": step, "best": best, "gate_thresholds": GATE,
                                                          "wall_s": elapsed_before + (time.time() - t_start), **m}, indent=2))

    step = start_step
    stop_reason = "done"
    K = max(args.rollout_steps, 0)
    while step < args.steps:
        if args.stop_at_step and step >= args.stop_at_step:
            stop_reason = f"stop_at_step {args.stop_at_step}"; break
        if args.max_minutes and (time.time() - t_main) / 60.0 > args.max_minutes:
            stop_reason = f"max_minutes {args.max_minutes}"; break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = train.sample(rng, args.batch, device, extra_steps=max(K, 1), domain_frac=args.domain_frac)
        loss, parts = step_loss(model, batch, context, loss_w, args.hold_bad_weight)
        if K > 0:
            ro, ro_parts = rollout_loss(model, batch, context, K, z1_mean_t, z1_std_t, loss_w, args.hold_bad_weight)
            loss = loss + args.rollout_weight * ro
            parts.update(ro_parts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        if step % 100 == 0 or step == start_step + 1:
            rec = {"phase": "train", "step": step, "loss": float(loss.detach()), **parts, "lr": lr_at(step - 1),
                   "sps": (step - start_step) * args.batch / max(time.time() - t_start, 1e-6)}
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if step % 1000 == 0 or step == start_step + 1:
                print(json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            vr = np.random.default_rng(args.seed + 1)
            with torch.no_grad():
                model.eval()
                vls = [float(step_loss(model, val.sample(vr, args.batch, device, extra_steps=1, domain_frac=0.5), context,
                                       loss_w, args.hold_bad_weight)[0]) for _ in range(args.val_batches)]
                model.train()
            m = evaluate(model, val, norm, context, args.val_steps, device, args.val_max_per_domain, args.seed + 2)
            m["val_loss"] = float(np.mean(vls))
            m["gpu_peak_gib"] = float(torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else 0.0
            m["train_sps"] = (step - start_step) * args.batch / max(time.time() - t_start, 1e-6)
            rec = {"phase": "val", "step": step, "val_loss": m["val_loss"], "sel": m["sel"], "gpu_peak_gib": m["gpu_peak_gib"],
                   "train_sps": m["train_sps"], **{k: v for k, v in m.items() if k.startswith("gate/")}}
            with log.open("a") as fh:
                fh.write(json.dumps({"phase": "val", "step": step, **m}) + "\n")
            print(json.dumps(rec), flush=True)
            print(gate_report(m), flush=True)
            last_metrics = m
            if m["sel"] < best["metric"]:
                best = {"metric": float(m["sel"]), "step": step}
                save(out_dir / "ckpt_best.pt", step, m, with_state=False)
            save(out_dir / "ckpt_last.pt", step, m, with_state=True)
            write_metrics(step, m)
            last_ckpt = time.time()
        elif (time.time() - last_ckpt) / 60.0 >= args.ckpt_every_min:
            save(out_dir / "ckpt_last.pt", step, last_metrics, with_state=True)
            last_ckpt = time.time()
    save(out_dir / "ckpt_last.pt", step, last_metrics, with_state=True)
    if last_metrics:
        write_metrics(step, last_metrics)
    (out_dir / "run_state.json").write_text(json.dumps(
        {"step": step, "steps": args.steps, "stop_reason": stop_reason, "best": best,
         "wall_s": elapsed_before + (time.time() - t_start), "load_s": t_start - t_main, "process_wall_s": time.time() - t_main,
         "finished": step >= args.steps}, indent=2))
    print(f"{stop_reason}: step {step}/{args.steps}, best sel {best['metric']:.4f} @ {best['step']}, "
          f"wall {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
