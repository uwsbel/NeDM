"""Where in the encoder does localisation die? Synthetic-data probe curve.

Answers open-questions.md ("Does a global pooled z2 survive at 256²?") as a
measurement rather than a geometric argument: fit a linear probe to recover the
vehicle's (x, y) from each pyramid stage and from the pooled z2, and report the
error in PIXELS.

Synthetic data isolates ARCHITECTURAL capacity from data quality. A bad number
here means the architecture cannot represent the signal, not that the frames
were hard. It cannot tell us whether a real encoder trained on real frames will
FIND the signal — only whether it is there to be found.
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
VEH_L, VEH_W = 15.0, 7.0  # open-questions.md: vehicle is ~15x7 px
ROCK_MIN, ROCK_MAX = 1.5, 3.0  # radii -> 3-6 px across
MARGIN = 20.0  # keep the vehicle clear of the border


def make_batch(bs: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 4, 256, 256) frames and (B, 3) targets [x_px, y_px, yaw_rad]."""
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, device=DEV, dtype=torch.float32),
        torch.arange(SIZE, device=DEV, dtype=torch.float32),
        indexing="ij",
    )

    # Low-frequency textured background, independent per channel.
    coarse = torch.rand(bs, 4, 16, 16, device=DEV, generator=gen)
    img = F.interpolate(coarse, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    img = 0.25 + 0.35 * img

    # 3-8 small distractor rocks per frame.
    n_rocks = 8
    rx = torch.rand(bs, n_rocks, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    ry = torch.rand(bs, n_rocks, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    rr = ROCK_MIN + torch.rand(bs, n_rocks, device=DEV, generator=gen) * (ROCK_MAX - ROCK_MIN)
    keep = (torch.rand(bs, n_rocks, device=DEV, generator=gen) > 0.35).float()
    for k in range(n_rocks):
        d = (xs - rx[:, k, None, None]) ** 2 + (ys - ry[:, k, None, None]) ** 2
        m = (d <= rr[:, k, None, None] ** 2).float() * keep[:, k, None, None]
        img = img * (1 - m.unsqueeze(1)) + m.unsqueeze(1) * 0.75

    # The vehicle: an oriented 15x7 px rectangle, brighter and colour-distinct.
    vx = torch.rand(bs, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    vy = torch.rand(bs, device=DEV, generator=gen) * (SIZE - 2 * MARGIN) + MARGIN
    yaw = torch.rand(bs, device=DEV, generator=gen) * 2 * torch.pi
    dx, dy = xs - vx[:, None, None], ys - vy[:, None, None]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    u, v = dx * c + dy * s, -dx * s + dy * c
    veh = ((u.abs() <= VEH_L / 2) & (v.abs() <= VEH_W / 2)).float().unsqueeze(1)
    colour = torch.tensor([0.95, 0.35, 0.10, 0.60], device=DEV).view(1, 4, 1, 1)
    img = img * (1 - veh) + veh * colour

    img = img + 0.02 * torch.randn(img.shape, device=DEV, generator=gen)
    return img.clamp(0, 1), torch.stack([vx, vy, yaw], dim=1)


class SoftArgmaxProbe(nn.Module):
    """1x1 conv -> spatial softmax -> expected (x, y), in [0, 1] image coords.

    Chosen over a linear probe on the flattened map because flattened dimension
    spans 128 to 524,288 across the stages, so a linear probe measures its own
    sample efficiency as much as the representation. This readout has C+1
    parameters at every stage, respects spatial structure, and its accuracy is
    limited by the stage's own resolution — which is the quantity in question.
    """

    def __init__(self, channels: int, grid: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)
        coord = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
        self.register_buffer("cx", coord.view(1, 1, grid).expand(1, grid, grid).reshape(1, -1))
        self.register_buffer("cy", coord.view(1, grid, 1).expand(1, grid, grid).reshape(1, -1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(feat).flatten(1), dim=1)
        return torch.stack([(w * self.cx).sum(1), (w * self.cy).sum(1)], dim=1)


class LinearProbe(nn.Module):
    """For the pooled latent, which has no spatial structure to soft-argmax."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_features, 256), nn.SiLU(), nn.Linear(256, 2))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat.flatten(1))


def build_probes(enc: ConvEncoderRGBD) -> tuple[list[str], list[nn.Module]]:
    with torch.no_grad():
        maps = enc.encode_pyramid(torch.zeros(1, 4, SIZE, SIZE, device=DEV))
        z2 = enc(torch.zeros(1, 4, SIZE, SIZE, device=DEV))
    names, probes = [], []
    for i, m in enumerate(maps, 1):
        names.append(f"stage{i} {tuple(m.shape[1:])}")
        probes.append(SoftArgmaxProbe(m.shape[1], m.shape[-1]).to(DEV))
    names.append(f"pooled z2 ({z2.shape[1]},)")
    probes.append(LinearProbe(z2.shape[1]).to(DEV))
    return names, probes


def reps(enc: ConvEncoderRGBD, x: torch.Tensor) -> list[torch.Tensor]:
    maps = enc.encode_pyramid(x)
    return maps + [enc.head(maps[-1])]


def run(enc: ConvEncoderRGBD, label: str, steps: int = 2000, bs: int = 64) -> None:
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    names, probes = build_probes(enc)
    opts = [torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=1e-5) for p in probes]
    gen = torch.Generator(device=DEV).manual_seed(1234)
    train_loss = [float("nan")] * len(probes)

    for _ in range(steps):
        x, tgt = make_batch(bs, gen)
        with torch.no_grad():
            rs = reps(enc, x)
        for i, (r, probe, opt) in enumerate(zip(rs, probes, opts)):
            loss = F.mse_loss(probe(r), tgt[:, :2] / (SIZE - 1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            train_loss[i] = loss.item()

    # Held-out evaluation, and a chance baseline that predicts the mean position.
    gen_test = torch.Generator(device=DEV).manual_seed(9999)
    errs = [[] for _ in probes]
    chance = []
    for _ in range(16):
        x, tgt = make_batch(bs, gen_test)
        with torch.no_grad():
            rs = reps(enc, x)
            for i, (r, probe) in enumerate(zip(rs, probes)):
                errs[i].append(((probe(r) * (SIZE - 1)) - tgt[:, :2]).norm(dim=1))
            mean_pred = torch.full_like(tgt[:, :2], (SIZE) / 2.0)
            chance.append((mean_pred - tgt[:, :2]).norm(dim=1))

    print(f"\n=== {label} ===")
    print(f"{'representation':>26} {'px/cell':>8} {'median_err_px':>14} {'p90_err_px':>11} {'final_train_mse':>16}")
    for (name, e), tl in zip(zip(names, errs), train_loss):
        e = torch.cat(e)
        cell = SIZE // int(name.split("(")[1].split(",")[-2].strip().rstrip(")")) if "stage" in name else "-"
        print(f"{name:>26} {str(cell):>8} {e.median().item():>14.2f} {e.quantile(0.9).item():>11.2f} {tl:>16.5f}")
    c = torch.cat(chance)
    print(f"{'chance (predict centre)':>26} {'-':>8} {c.median().item():>14.2f} {c.quantile(0.9).item():>11.2f} {'-':>16}")


def brief_train(enc: ConvEncoderRGBD, steps: int = 400, bs: int = 32) -> ConvEncoderRGBD:
    """Short RGB reconstruction warm-up: gives the trained-encoder arm."""
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
    x, tgt = make_batch(4, torch.Generator(device=DEV).manual_seed(7))
    print("sanity: frames", tuple(x.shape), "targets", tuple(tgt.shape))
    print("vehicle occupies", f"{(VEH_L*VEH_W)/(SIZE*SIZE)*100:.3f}% of pixels")

    run(ConvEncoderRGBD().to(DEV), "RANDOM-INIT ENCODER (information floor)")
    run(brief_train(ConvEncoderRGBD().to(DEV)), "BRIEFLY-TRAINED ENCODER (AE warm-up)")
