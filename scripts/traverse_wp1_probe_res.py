"""Fallback-sizing probe: how small can the planner-facing spatial map be?

Plan §5's pre-declared fallback keeps a low-res spatial feature map as the
planner-facing representation. This script freezes a trained WP1 encoder and
trains BEV-occupancy probes on avg-pooled / channel-projected versions of the
pre-pooling map, reporting held-out-layout IoU per (resolution, channels)
variant — the measured size/quality frontier for the fallback decision.

  PYTHONPATH=src python scripts/traverse_wp1_probe_res.py --ckpt artifacts/traverse/wp1_v3/ckpt_warmup.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nedm.traverse import perception as P  # noqa: E402
from traverse_wp1_train import cycle, make_loader, to_device  # noqa: E402

VARIANTS = [(16, 256), (8, 256), (8, 64), (4, 256), (4, 64)]  # (res, channels)


class ResProbe(nn.Module):
    def __init__(self, res: int, ch: int, c_map: int = 256):
        super().__init__()
        self.res = res
        self.proj = nn.Identity() if ch == c_map else nn.Conv2d(c_map, ch, 1)
        layers: list[nn.Module] = []
        c, cur, c_next = ch, res, 128
        while cur < P.BEV_GRID:
            layers += [P._conv_block(c, c_next), nn.Upsample(scale_factor=2)]
            c, c_next, cur = c_next, max(32, c_next // 2), cur * 2
        layers += [P._conv_block(c, 32), nn.Conv2d(32, 1, 1)]
        self.dec = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.proj(F.adaptive_avg_pool2d(s, self.res))
        return self.dec(x)[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/traverse/wp1_v3/ckpt_warmup.pt")
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
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-batches", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--variants", nargs="+", default=None, help="ResXChannels, e.g. 16x64 8x128")
    ap.add_argument("--out-name", default="probe_res_readout.json")
    args = ap.parse_args()
    variants = (
        [tuple(int(x) for x in v.split("x")) for v in args.variants] if args.variants else VARIANTS
    )

    device = "cuda"
    torch.manual_seed(args.seed)
    roots = [Path(r) for r in args.data_roots]
    train_e, val_e, _ = P.split_episodes(roots, seed=args.seed)
    train_ds = P.WP1FrameDataset(train_e, Path(args.arena))
    val_ds = P.WP1FrameDataset(val_e, Path(args.arena))
    train_loader = make_loader(train_ds, args.batch, args.workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch, 4, shuffle=True)

    encoder = P.Encoder().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    probes = {f"{r}x{r}x{c}": ResProbe(r, c).to(device) for r, c in variants}
    opt = torch.optim.AdamW([p for m in probes.values() for p in m.parameters()], lr=3e-4)
    pw = torch.tensor(5.0, device=device)

    it = cycle(train_loader)
    for step in range(1, args.steps + 1):
        batch = to_device(next(it), device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _, s = encoder(batch["input"])
        with torch.autocast("cuda", torch.bfloat16):
            loss = sum(
                F.binary_cross_entropy_with_logits(m(s.detach()), batch["bev"], pos_weight=pw)
                for m in probes.values()
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print({"step": step, "loss": float(loss.item())}, flush=True)

    for m in probes.values():
        m.eval()
    inter = {k: 0.0 for k in probes}
    union = {k: 0.0 for k in probes}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= args.val_batches:
                break
            n += 1
            batch = to_device(batch, device)
            with torch.autocast("cuda", torch.bfloat16):
                _, s = encoder(batch["input"])
                for k, m in probes.items():
                    i, u = P.bev_counts(m(s).float(), batch["bev"])
                    inter[k] += i
                    union[k] += u

    readout = {
        "ckpt": args.ckpt,
        "val_batches": n,
        "bev_iou_by_variant": {k: inter[k] / max(union[k], 1) for k in probes},
        "floats_per_frame": {f"{r}x{r}x{c}": r * r * c for r, c in variants},
    }
    out_path = Path(args.ckpt).parent / args.out_name
    out_path.write_text(json.dumps(readout, indent=2))
    print(json.dumps(readout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
