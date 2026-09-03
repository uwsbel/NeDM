"""Occupancy/class-mask arm of the WP1 probe curve.

Companion to wp1_probe_resolution.py. Same synthetic generator, same pyramid,
same random-init and briefly-trained arms, same held-out discipline — but the
target is a per-pixel class mask rather than a point coordinate.

The question this answers that localisation could not: WHICH information does
pooling destroy? Occupancy is a lower-frequency quantity than a 15x7 px
position, so if it survives pooling where localisation did not, the plan's
global z2 may be adequate for one of its two jobs and not the other.

Same caveat as the localisation arm: synthetic masks say the signal is there to
be found, not that a real encoder on real frames will find it.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/kyle/sbel/NeDM/src")
from nedm.nrd.vision_rgbd import ConvEncoderRGBD, ConvDecoderRGBD  # noqa: E402

DEV = "cuda"
SIZE = 256
MASK = 64  # evaluation resolution: vehicle spans ~3.75 x 1.75 cells here
VEH_L, VEH_W = 15.0, 7.0
ROCK_MIN, ROCK_MAX = 1.5, 3.0
MARGIN = 20.0
CLASSES = ("background", "rock", "vehicle")
N_CLASSES = len(CLASSES)


def make_batch(bs: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 4, 256, 256) frames and (B, 64, 64) class-index masks."""
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, device=DEV, dtype=torch.float32),
        torch.arange(SIZE, device=DEV, dtype=torch.float32),
        indexing="ij",
    )
    coarse = torch.rand(bs, 4, 16, 16, device=DEV, generator=gen)
    img = 0.25 + 0.35 * F.interpolate(coarse, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    cls = torch.zeros(bs, SIZE, SIZE, device=DEV)

    n_rocks = 8
    rx = torch.rand(bs, n_rocks, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    ry = torch.rand(bs, n_rocks, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    rr = ROCK_MIN + torch.rand(bs, n_rocks, device=DEV, generator=gen) * (ROCK_MAX - ROCK_MIN)
    keep = (torch.rand(bs, n_rocks, device=DEV, generator=gen) > 0.35).float()
    for k in range(n_rocks):
        d = (xs - rx[:, k, None, None]) ** 2 + (ys - ry[:, k, None, None]) ** 2
        m = (d <= rr[:, k, None, None] ** 2).float() * keep[:, k, None, None]
        img = img * (1 - m.unsqueeze(1)) + m.unsqueeze(1) * 0.75
        cls = torch.maximum(cls, m * 1.0)

    vx = torch.rand(bs, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    vy = torch.rand(bs, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    yaw = torch.rand(bs, device=DEV, generator=gen) * 2 * torch.pi
    dx, dy = xs - vx[:, None, None], ys - vy[:, None, None]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    u, v = dx * c + dy * s, -dx * s + dy * c
    veh = ((u.abs() <= VEH_L / 2) & (v.abs() <= VEH_W / 2)).float()
    colour = torch.tensor([0.95, 0.35, 0.10, 0.60], device=DEV).view(1, 4, 1, 1)
    img = img * (1 - veh.unsqueeze(1)) + veh.unsqueeze(1) * colour
    cls = torch.maximum(cls, veh * 2.0)

    img = (img + 0.02 * torch.randn(img.shape, device=DEV, generator=gen)).clamp(0, 1)
    # Max-pool the class map so small objects survive downsampling; class index
    # order is deliberately background < rock < vehicle so max = priority.
    cls_small = F.max_pool2d(cls.unsqueeze(1), kernel_size=SIZE // MASK).squeeze(1)
    return img, cls_small.long()


class ConvMaskProbe(nn.Module):
    """1x1 conv at the stage's own resolution, then bilinear to MASK.

    C * N_CLASSES + N_CLASSES parameters, so capacity is comparable across
    stages and the resolution limit is the stage's own, which is the quantity
    under test.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, N_CLASSES, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.score(feat), size=(MASK, MASK), mode="bilinear", align_corners=False)


class LatentMaskProbe(nn.Module):
    """Pooled z2 has no spatial structure, so it gets an MLP to the full mask.

    This gives z2 ~1.6 M parameters against the conv probes' few thousand. The
    asymmetry favours z2; if it still loses, the loss is not a capacity artefact.
    """

    def __init__(self, z2_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z2_dim, 512), nn.SiLU(), nn.Linear(512, N_CLASSES * MASK * MASK)
        )

    def forward(self, z2: torch.Tensor) -> torch.Tensor:
        return self.net(z2).view(-1, N_CLASSES, MASK, MASK)


def reps(enc: ConvEncoderRGBD, x: torch.Tensor) -> list[torch.Tensor]:
    maps = enc.encode_pyramid(x)
    return maps + [enc.head(maps[-1])]


def iou_recall(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[list[float], list[float]]:
    ious, recalls = [], []
    for c in range(N_CLASSES):
        p, t = pred == c, tgt == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(inter / union if union else float("nan"))
        recalls.append(inter / t.sum().item() if t.sum() else float("nan"))
    return ious, recalls


def run(enc: ConvEncoderRGBD, label: str, steps: int = 2000, bs: int = 64) -> None:
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        rs = reps(enc, torch.zeros(1, 4, SIZE, SIZE, device=DEV))
    names = [f"stage{i} {tuple(r.shape[1:])}" for i, r in enumerate(rs[:-1], 1)] + [
        f"pooled z2 ({rs[-1].shape[1]},)"
    ]
    probes = [ConvMaskProbe(r.shape[1]).to(DEV) for r in rs[:-1]] + [
        LatentMaskProbe(rs[-1].shape[1]).to(DEV)
    ]
    opts = [torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=1e-5) for p in probes]

    # Rare classes dominate nothing without this: vehicle is ~0.16% of pixels.
    weight = torch.tensor([1.0, 20.0, 40.0], device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(1234)
    for _ in range(steps):
        x, m = make_batch(bs, gen)
        with torch.no_grad():
            rr = reps(enc, x)
        for r, probe, opt in zip(rr, probes, opts):
            loss = F.cross_entropy(probe(r), m, weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    gen_test = torch.Generator(device=DEV).manual_seed(9999)
    acc_i = [[0.0] * N_CLASSES for _ in probes]
    acc_r = [[0.0] * N_CLASSES for _ in probes]
    n_batches = 16
    for _ in range(n_batches):
        x, m = make_batch(bs, gen_test)
        with torch.no_grad():
            rr = reps(enc, x)
            for i, (r, probe) in enumerate(zip(rr, probes)):
                ious, recs = iou_recall(probe(r).argmax(1), m)
                for c in range(N_CLASSES):
                    acc_i[i][c] += ious[c] / n_batches
                    acc_r[i][c] += recs[c] / n_batches

    print(f"\n=== {label} ===")
    hdr = f"{'representation':>26} {'px/cell':>8}"
    for c in CLASSES:
        hdr += f" {c[:4]+'_IoU':>10}"
    hdr += f" {'veh_recall':>11} {'rock_recall':>12}"
    print(hdr)
    for name, ii, rr_ in zip(names, acc_i, acc_r):
        cell = SIZE // int(name.split(",")[-1].strip().rstrip(")")) if "stage" in name else "-"
        line = f"{name:>26} {str(cell):>8}"
        for c in range(N_CLASSES):
            line += f" {ii[c]:>10.3f}"
        line += f" {rr_[2]:>11.3f} {rr_[1]:>12.3f}"
        print(line)
    # Trivial baseline: predict background everywhere.
    x, m = make_batch(bs, torch.Generator(device=DEV).manual_seed(5555))
    ii, rr_ = iou_recall(torch.zeros_like(m), m)
    print(f"{'all-background baseline':>26} {'-':>8} {ii[0]:>10.3f} {ii[1]:>10.3f} {ii[2]:>10.3f} {rr_[2]:>11.3f} {rr_[1]:>12.3f}")
    frac = [(m == c).float().mean().item() for c in range(N_CLASSES)]
    print(f"  class pixel fractions at {MASK}x{MASK}: " + ", ".join(f"{c}={f:.4f}" for c, f in zip(CLASSES, frac)))


def brief_train(enc: ConvEncoderRGBD, steps: int = 400, bs: int = 32) -> ConvEncoderRGBD:
    dec = ConvDecoderRGBD(z2_dim=enc.z2_dim, out_channels=4).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)
    gen = torch.Generator(device=DEV).manual_seed(4321)
    enc.train()
    for _ in range(steps):
        x, _ = make_batch(bs, gen)
        loss = F.mse_loss(dec(enc(x)), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"[brief AE warm-up] final reconstruction MSE = {loss.item():.5f}")
    return enc


if __name__ == "__main__":
    torch.manual_seed(0)
    run(ConvEncoderRGBD().to(DEV), "RANDOM-INIT ENCODER (information floor)")
    run(brief_train(ConvEncoderRGBD().to(DEV)), "BRIEFLY-TRAINED ENCODER (AE warm-up)")
