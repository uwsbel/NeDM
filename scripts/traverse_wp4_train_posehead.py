#!/usr/bin/env python
"""Camera-based vehicle localisation (WP4 step 3): per-frame vehicle pose from the frozen
camera encoder's 64x64 stage-2 map.

The tracker's pose in the Chrono evaluation is still the simulator's (the last privileged
input on the tracking side). WP1's probe put the vehicle at 0.8 m / 3-4 deg from the 16x16
map; here the 64x64 stage-2 map (1.36 m/cell) is decoded with a heatmap + soft-argmax +
sub-cell regression head, which is the standard recipe for sub-cell keypoint accuracy.

Input   4 x 256 x 256 RGB + normalized elevation (the WP1 frame), frozen encoder stem
Output  vehicle centre (u, v) in image pixels, yaw (sin, cos)
Labels  recorded pose from states.npz (privileged, training only)
Metric  centre error in metres after pixel -> world with the known arena heightmap
        (fixed terrain = the 'memorized terrain' rung), yaw error in degrees.

Usage (cluster):  PYTHONPATH=src python scripts/traverse_wp4_train_posehead.py --out artifacts/traverse/wp4_posehead_v1_amd \
    --episode-manifest artifacts/traverse/wp1_v6_episode_manifest.json --steps 15000 --workers 20
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse import perception as P
from nedm.traverse.camera import CameraModel
from nedm.traverse.masks import VEHICLE_HGT_M
from nedm.traverse.terrain import TerrainMap

STAGE = 4          # encoder.backbone[:4] -> 64 ch x 64 x 64
CELL_PX = 4.0      # image px per stage cell
HEAT_SIGMA = 1.0   # stage cells


class PoseFrames(P.WP1FrameDataset):
    """WP1 frame without the label/BEV rasterization: input + pose only."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, t = self._locate(int(idx))
        reader = self._reader(ep_i)
        win = reader.read_window(t, 1)
        if self._field_idx is None:
            self._field_idx = {name: i for i, name in enumerate(win["state_fields"])}
        fi = self._field_idx
        state = win["states"][0]
        x, y, yaw = float(state[fi["pos_x_m"]]), float(state[fi["pos_y_m"]]), float(state[fi["yaw_rad"]])
        gz = float(self.tmap.height(x, y))
        rgb = win["rgb"][0].astype(np.float32) / 255.0
        z_norm = self._z_map(win["depth_mm"][0])
        u, v = self.cam.world_to_pixel(x, y, gz + 0.5 * VEHICLE_HGT_M)
        inp = np.concatenate([rgb.transpose(2, 0, 1), z_norm[None]], axis=0)
        return {"input": torch.from_numpy(inp), "uv": torch.tensor([float(u), float(v)]),
                "xy": torch.tensor([x, y]), "yaw": torch.tensor([math.sin(yaw), math.cos(yaw)]),
                "speed": torch.tensor(float(state[fi["vel_body_x_mps"]]))}


class PoseHead(nn.Module):
    def __init__(self, c_in: int = 64, width: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(c_in, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU())
        self.heat = nn.Conv2d(width, 1, 1)
        self.reg = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 4))
        g = torch.arange(64, dtype=torch.float32)
        self.register_buffer("gu", g.view(1, 1, 64).expand(1, 64, 64).reshape(1, -1))
        self.register_buffer("gv", g.view(1, 64, 1).expand(1, 64, 64).reshape(1, -1))

    def forward(self, fmap: torch.Tensor):
        f = self.trunk(fmap)
        logits = self.heat(f)[:, 0]                                  # (B, 64, 64), rows = v, cols = u
        prob = torch.softmax(logits.flatten(1), dim=1)
        u = (prob * self.gu).sum(1); v = (prob * self.gv).sum(1)     # soft-argmax, stage cells
        grid = torch.stack([(u + 0.5) * 2 / 64 - 1, (v + 0.5) * 2 / 64 - 1], -1).view(-1, 1, 1, 2)
        feat = F.grid_sample(f, grid, mode="bilinear", align_corners=False)[:, :, 0, 0]
        out = self.reg(feat)
        du, dv = out[:, 0], out[:, 1]
        yaw = F.normalize(out[:, 2:4], dim=1)
        return logits, u + du, v + dv, yaw


def stage_to_img(u_s, v_s):
    return u_s * CELL_PX + 1.5, v_s * CELL_PX + 1.5


def img_to_stage(u, v):
    return (u - 1.5) / CELL_PX, (v - 1.5) / CELL_PX


def pixel_to_world(cam: CameraModel, tmap: TerrainMap, u: np.ndarray, v: np.ndarray, iters: int = 3):
    """Invert the pinhole projection at the vehicle-centre height using the known arena heightmap."""
    z = np.zeros_like(u)
    for _ in range(iters):
        scale = cam.f_px / (cam.cam_height_m - z)
        x, y = (u - cam.cx) / scale, (cam.cy - v) / scale
        z = tmap.height(np.clip(x, -tmap.half, tmap.half), np.clip(y, -tmap.half, tmap.half)) + 0.5 * VEHICLE_HGT_M
    return x, y


@torch.no_grad()
def evaluate(stem, head, loader, device, cam, tmap, max_batches):
    xy_err, yaw_err, moving = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        fmap = stem(batch["input"].to(device))
        _, u_s, v_s, yaw = head(fmap)
        u, v = stage_to_img(u_s.cpu().numpy(), v_s.cpu().numpy())
        x, y = pixel_to_world(cam, tmap, u, v)
        xy_err.append(np.hypot(x - batch["xy"][:, 0].numpy(), y - batch["xy"][:, 1].numpy()))
        yp = torch.atan2(yaw[:, 0], yaw[:, 1]).cpu().numpy(); yt = torch.atan2(batch["yaw"][:, 0], batch["yaw"][:, 1]).numpy()
        yaw_err.append(np.degrees(np.abs((yp - yt + np.pi) % (2 * np.pi) - np.pi)))
        moving.append(batch["speed"].numpy() > 0.5)
    xy_err, yaw_err, moving = np.concatenate(xy_err), np.concatenate(yaw_err), np.concatenate(moving)
    return {"frames": int(len(xy_err)), "xy_mean_m": float(xy_err.mean()), "xy_median_m": float(np.median(xy_err)),
            "xy_p95_m": float(np.percentile(xy_err, 95)), "yaw_mean_deg": float(yaw_err.mean()),
            "yaw_p95_deg": float(np.percentile(yaw_err, 95)),
            "xy_mean_moving_m": float(xy_err[moving].mean()) if moving.any() else None,
            "yaw_mean_moving_deg": float(yaw_err[moving].mean()) if moving.any() else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-roots", nargs="+", default=["artifacts/traverse/pilot_v1", "artifacts/traverse/full_v1",
                    "artifacts/traverse/full_v2", "artifacts/traverse/full_v3", "artifacts/traverse/full_v4_partial"])
    ap.add_argument("--episode-manifest", default=None)
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--encoder-ckpt", default="artifacts/traverse/wp1_v6/ckpt_warmup.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--final-val-batches", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    roots = [Path(r) for r in args.data_roots]
    manifest = json.loads(Path(args.episode_manifest).read_text()) if args.episode_manifest else None
    train_e, val_e, _ = P.split_episodes(roots, seed=args.seed, manifest=manifest)
    print(f"episodes train {len(train_e)} val {len(val_e)} (test untouched)", flush=True)
    arena = Path(args.arena)
    tmap, cam = TerrainMap.from_dir(arena), CameraModel()
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                                        pin_memory=True, drop_last=shuffle, persistent_workers=args.workers > 0,
                                        prefetch_factor=4 if args.workers else None)
    train_loader, val_loader = mk(PoseFrames(train_e, arena), True), mk(PoseFrames(val_e, arena), False)

    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder = P.Encoder(z_dim=256, n_q=8).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=True); encoder.eval()
    stem = encoder.backbone[:STAGE]
    for p in stem.parameters():
        p.requires_grad_(False)
    head = PoseHead(width=args.width).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    gu = head.gu.view(64, 64); gv = head.gv.view(64, 64)
    log = (out / "train_log.jsonl").open("w")
    best, step, t0 = math.inf, 0, time.time()
    it = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        step += 1
        with torch.no_grad():
            fmap = stem(batch["input"].to(device, non_blocking=True))
        u_t, v_t = img_to_stage(batch["uv"][:, 0].to(device), batch["uv"][:, 1].to(device))
        yaw_t = batch["yaw"].to(device)
        logits, u_p, v_p, yaw_p = head(fmap)
        target = torch.exp(-((gu[None] - u_t[:, None, None]) ** 2 + (gv[None] - v_t[:, None, None]) ** 2) / (2 * HEAT_SIGMA ** 2))
        target = target.flatten(1); target = target / target.sum(1, keepdim=True)
        ce = -(target * F.log_softmax(logits.flatten(1), dim=1)).sum(1).mean()
        l_uv = (F.smooth_l1_loss(u_p, u_t) + F.smooth_l1_loss(v_p, v_t))
        l_yaw = F.l1_loss(yaw_p, yaw_t)
        loss = ce + l_uv + 2.0 * l_yaw
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "ce": ce.item(), "uv": l_uv.item(), "yaw": l_yaw.item(),
                   "sps": step * args.batch / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 500 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            head.eval()
            ev = evaluate(stem, head, val_loader, device, cam, tmap,
                          args.final_val_batches if step == args.steps else args.val_batches)
            head.train()
            ev["step"] = step
            print(json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n"); log.flush()
            if ev["xy_mean_m"] < best:
                best = ev["xy_mean_m"]
                torch.save({"head": head.state_dict(), "config": vars(args), "eval": ev, "step": step,
                            "encoder_ckpt": args.encoder_ckpt}, out / "ckpt_best.pt")
                (out / "readout.json").write_text(json.dumps(ev, indent=1))
    print(f"done; best val xy {best:.3f} m; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
