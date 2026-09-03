"""WP1 perception pilot trainer (plan §5, gate G1 provisional numbers).

Phase 1 (warm-up): encoder + mandatory auxiliary heads on z2 — class masks,
vehicle-center heatmap + yaw, foreground-weighted RGB recon, elevation recon,
BEV occupancy. Phase 2 (probes): encoder frozen, capacity-matched probes
decode BEV occupancy + vehicle pose from z2 AND from the pre-pooling spatial
map; the gap is the measured cost of global pooling (§5 staging decision).

Split is layout-level 70/15/15; metrics reported on held-out-layout val
episodes; the test split is recorded but untouched (plan §12.1).

  PYTHONPATH=src python scripts/traverse_wp1_train.py \
      --data-roots artifacts/traverse/pilot_v1 artifacts/traverse/full_v1
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
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nedm.traverse import perception as P  # noqa: E402


def make_loader(dataset, batch: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def cycle(loader):
    while True:
        yield from loader


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(encoder, heads, sp_heads, probes, loader, device, half_size_m, max_batches):
    encoder.eval(), heads.eval()
    if sp_heads is not None:
        sp_heads.eval()
    agg = {k: np.zeros(P.N_CLASSES) for k in ("inter", "union", "target", "correct")}
    agg_sp = {k: np.zeros(P.N_CLASSES) for k in ("inter", "union", "target", "correct")}
    bev_i = {"warmup": 0.0, "z2": 0.0, "spatial": 0.0, "sphead": 0.0}
    bev_u = {"warmup": 0.0, "z2": 0.0, "spatial": 0.0, "sphead": 0.0}
    xy_err = {"warmup": [], "z2": [], "spatial": []}
    yaw_err = {"warmup": [], "z2": [], "spatial": []}
    n = 0
    for batch in loader:
        if n >= max_batches:
            break
        n += 1
        batch = to_device(batch, device)
        with torch.autocast("cuda", torch.bfloat16):
            z, s = encoder(batch["input"])
            out = heads(z)
            sp_out = sp_heads(s) if sp_heads is not None else None
        m = P.seg_metrics(out["seg"].float(), batch["label"])
        for k in agg:
            agg[k] += m[k]
        i, u = P.bev_counts(out["bev"].float(), batch["bev"])
        bev_i["warmup"] += i
        bev_u["warmup"] += u
        xe, ye = P.pose_errors(out["pose"].float(), batch, half_size_m)
        xy_err["warmup"].append(xe)
        yaw_err["warmup"].append(ye)
        if sp_out is not None:
            msp = P.seg_metrics(sp_out["seg"].float(), batch["label"])
            for k in agg_sp:
                agg_sp[k] += msp[k]
            i, u = P.bev_counts(sp_out["bev"].float(), batch["bev"])
            bev_i["sphead"] += i
            bev_u["sphead"] += u
        if probes is not None:
            z2_probe, sp_probe = probes
            z2_probe.eval(), sp_probe.eval()
            with torch.autocast("cuda", torch.bfloat16):
                bl_z, pose_z = z2_probe(z)
                bl_s, pose_s = sp_probe(s)
            for name, bl, pose in (("z2", bl_z, pose_z), ("spatial", bl_s, pose_s)):
                i, u = P.bev_counts(bl.float(), batch["bev"])
                bev_i[name] += i
                bev_u[name] += u
                xe, ye = P.pose_errors(pose.float(), batch, half_size_m)
                xy_err[name].append(xe)
                yaw_err[name].append(ye)
    encoder.train(), heads.train()

    iou = agg["inter"] / np.maximum(agg["union"], 1)
    recall = agg["correct"] / np.maximum(agg["target"], 1)
    out = {
        "seg_iou": {c: float(iou[c]) for c in range(P.N_CLASSES)},
        "seg_recall": {c: float(recall[c]) for c in range(P.N_CLASSES)},
        "bev_iou_warmup": float(bev_i["warmup"] / max(bev_u["warmup"], 1)),
    }
    if sp_heads is not None:
        iou_sp = agg_sp["inter"] / np.maximum(agg_sp["union"], 1)
        rec_sp = agg_sp["correct"] / np.maximum(agg_sp["target"], 1)
        out["seg_iou_sphead"] = {c: float(iou_sp[c]) for c in range(P.N_CLASSES)}
        out["seg_recall_sphead"] = {c: float(rec_sp[c]) for c in range(P.N_CLASSES)}
        out["bev_iou_sphead"] = float(bev_i["sphead"] / max(bev_u["sphead"], 1))
        sp_heads.train()
    for name in ("warmup", "z2", "spatial"):
        if not xy_err[name]:
            continue
        xe = np.concatenate(xy_err[name])
        ye = np.concatenate(yaw_err[name])
        if name != "warmup":
            out[f"bev_iou_{name}"] = float(bev_i[name] / max(bev_u[name], 1))
        out[f"center_err_m_{name}"] = {"mean": float(xe.mean()), "p95": float(np.percentile(xe, 95))}
        out[f"yaw_err_deg_{name}"] = {"mean": float(ye.mean()), "p95": float(np.percentile(ye, 95))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-roots",
        nargs="+",
        default=[
            "artifacts/traverse/pilot_v1",
            "artifacts/traverse/full_v1",
            "artifacts/traverse/full_v2",
        ],
    )
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--out", default="artifacts/traverse/wp1_v1")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--probe-steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--final-val-batches", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    roots = [Path(r) for r in args.data_roots]
    train_e, val_e, test_e = P.split_episodes(roots, seed=args.seed)
    print(f"episodes: train {len(train_e)} / val {len(val_e)} / test {len(test_e)} (layout-level split)")
    train_ds = P.WP1FrameDataset(train_e, Path(args.arena))
    val_ds = P.WP1FrameDataset(val_e, Path(args.arena))
    half_size_m = train_ds.tmap.size_m / 2.0

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "episodes": {"train": len(train_e), "val": len(val_e), "test": len(test_e)},
                "test_episodes": [str(e.ep_dir) for e in test_e],
                "loss_weights": P.LOSS_WEIGHTS,
                "seg_class_weights": P.SEG_CLASS_WEIGHTS,
            },
            indent=2,
        )
    )

    train_loader = make_loader(train_ds, args.batch, args.workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch, max(2, args.workers // 2), shuffle=True)

    encoder = P.Encoder().to(device)
    heads = P.WarmupHeads().to(device)
    sp_heads = P.SpatialAuxHeads().to(device)
    params = list(encoder.parameters()) + list(heads.parameters()) + list(sp_heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ---- Phase 1: warm-up with mandatory auxiliary losses -----------------
    it = cycle(train_loader)
    t0 = time.time()
    best_score = -1.0
    with log_path.open("a", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            batch = to_device(next(it), device)
            with torch.autocast("cuda", torch.bfloat16):
                z, s = encoder(batch["input"])
                out = heads(z)
                sp_out = sp_heads(s)
                losses = P.warmup_losses(out, batch, half_size_m, sp_out)
                loss = sum(P.LOSS_WEIGHTS[k] * v for k, v in losses.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            if step % 100 == 0:
                rec = {
                    "phase": "warmup",
                    "step": step,
                    "loss": float(loss.item()),
                    **{k: float(v.item()) for k, v in losses.items()},
                    "sps": round(step * args.batch / (time.time() - t0), 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 1000 == 0:
                    print(rec, flush=True)
            if step % args.val_every == 0 or step == args.steps:
                m = evaluate(encoder, heads, sp_heads, None, val_loader, device, half_size_m, args.val_batches)
                m.update({"phase": "warmup_val", "step": step})
                log.write(json.dumps(m) + "\n")
                log.flush()
                print(m, flush=True)
                # Composite score over both decode paths: asset-class IoUs +
                # BEV IoUs (v1 selected on rock IoU alone, which was
                # noise-level, and froze an early checkpoint).
                score = (
                    float(np.mean([m["seg_iou"][c] for c in (1, 2, 3, 4)]))
                    + m["bev_iou_warmup"]
                    + float(np.mean([m["seg_iou_sphead"][c] for c in (1, 2, 3, 4)]))
                    + m["bev_iou_sphead"]
                )
                if score > best_score:
                    best_score = score
                    torch.save(
                        {
                            "encoder": encoder.state_dict(),
                            "heads": heads.state_dict(),
                            "sp_heads": sp_heads.state_dict(),
                            "step": step,
                        },
                        out_dir / "ckpt_warmup.pt",
                    )

    # ---- Phase 2: frozen-encoder probes (z2 vs spatial map) ---------------
    ckpt = torch.load(out_dir / "ckpt_warmup.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    heads.load_state_dict(ckpt["heads"])
    sp_heads.load_state_dict(ckpt["sp_heads"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    z2_probe = P.LatentProbe().to(device)
    sp_probe = P.SpatialProbe().to(device)
    popt = torch.optim.AdamW(list(z2_probe.parameters()) + list(sp_probe.parameters()), lr=args.lr)
    with log_path.open("a", encoding="utf-8") as log:
        for step in range(1, args.probe_steps + 1):
            batch = to_device(next(it), device)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                z, s = encoder(batch["input"])
            with torch.autocast("cuda", torch.bfloat16):
                bl_z, pose_z = z2_probe(z.detach())
                bl_s, pose_s = sp_probe(s.detach())
                loss = P.probe_losses(bl_z, pose_z, batch, half_size_m) + P.probe_losses(
                    bl_s, pose_s, batch, half_size_m
                )
            popt.zero_grad(set_to_none=True)
            loss.backward()
            popt.step()
            if step % 500 == 0:
                rec = {"phase": "probe", "step": step, "loss": float(loss.item())}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
    torch.save({"z2_probe": z2_probe.state_dict(), "spatial_probe": sp_probe.state_dict()}, out_dir / "ckpt_probes.pt")

    # ---- Final readout: held-out layouts AND training layouts -------------
    # Train-layout eval is the arch-vs-data diagnostic: good train + bad val
    # means generalization/data; bad train means the architecture or training
    # recipe cannot represent the task at all.
    class_names = {0: "background", 1: "rock", 2: "tree", 3: "house", 4: "vehicle"}

    def section(final: dict) -> dict:
        return {
            "seg_iou_from_z2": {class_names[c]: final["seg_iou"][c] for c in range(P.N_CLASSES)},
            "seg_recall_from_z2": {class_names[c]: final["seg_recall"][c] for c in range(P.N_CLASSES)},
            "seg_iou_from_spatial": {class_names[c]: final["seg_iou_sphead"][c] for c in range(P.N_CLASSES)},
            "seg_recall_from_spatial": {class_names[c]: final["seg_recall_sphead"][c] for c in range(P.N_CLASSES)},
            "bev_iou": {k: final[f"bev_iou_{k}"] for k in ("warmup", "z2", "spatial", "sphead")},
            "center_err_m": {k: final[f"center_err_m_{k}"] for k in ("warmup", "z2", "spatial")},
            "yaw_err_deg": {k: final[f"yaw_err_deg_{k}"] for k in ("warmup", "z2", "spatial")},
            "pooling_gap_bev_iou": final["bev_iou_spatial"] - final["bev_iou_z2"],
        }

    final_val = evaluate(
        encoder, heads, sp_heads, (z2_probe, sp_probe), val_loader, device, half_size_m, args.final_val_batches
    )
    train_eval_loader = make_loader(train_ds, args.batch, max(2, args.workers // 2), shuffle=True)
    final_train = evaluate(
        encoder, heads, sp_heads, (z2_probe, sp_probe), train_eval_loader, device, half_size_m, args.final_val_batches
    )
    readout = {
        "split": {"train": len(train_e), "val": len(val_e), "test_untouched": len(test_e)},
        "warmup_steps": args.steps,
        "probe_steps": args.probe_steps,
        "val_heldout_layouts": section(final_val),
        "train_layouts": section(final_train),
    }
    (out_dir / "g1_readout.json").write_text(json.dumps(readout, indent=2))
    print(json.dumps(readout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
