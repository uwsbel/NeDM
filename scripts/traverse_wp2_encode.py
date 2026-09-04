"""WP2 z2 precompute: frozen WP1 encoder -> per-episode latent cache.

Plan section 8.1 trains the joint NRD with a frozen encoder first, so every
run in the G3 triad reads exactly the same z2. Encoding once turns the dynamics
runs into table lookups: no zstd decode, no 4-channel conv, ~1 KB/frame instead
of ~0.3 MiB. One .npz per episode:

    z2    (T, z_dim) float32   frozen encoder latent
    z1    (T, 15)    float32   tire_normal_force_omega preset
    act   (T, 3)     float32   driver steering / throttle / braking
    pose  (T, 3)     float32   pos_x_m, pos_y_m, yaw_rad

pose is both the privileged-row input and the ground truth for dead-reckoned
rollout scoring; it is never an input to the state-only or joint models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nedm.traverse import perception as P
from nedm.traverse.camera import CameraModel
from nedm.traverse.storage import DEPTH_NO_HIT, DEPTH_OFFSET_M, EpisodeReader
from nedm.traverse.terrain import TerrainMap
from nedm.training.constants import (
    DEFAULT_ACTION_FIELDS,
    DEFAULT_ROLLOUT_FIELDS,
    STATE_FIELD_PRESETS,
)

STATE_FIELDS = STATE_FIELD_PRESETS["tire_normal_force_omega"]


def episode_key(ep_dir: Path) -> str:
    return f"{ep_dir.parent.name}__{ep_dir.name}"


class EpisodeFrames(Dataset):
    """One item = one whole episode. RGB stays uint8 and depth is converted to
    the normalized elevation image in the worker, so ~130 MB crosses the queue
    per episode instead of the 419 MB a float32 4-channel stack would cost."""

    def __init__(self, entries: list[P.EpisodeEntry], arena_dir: Path) -> None:
        self.entries = entries
        self.arena_dir = arena_dir
        self._cam: CameraModel | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def _camera(self) -> tuple[CameraModel, np.ndarray, float, float]:
        if self._cam is None:
            tmap = TerrainMap.from_dir(self.arena_dir)
            self._cam = CameraModel()
            _, _, sec = self._cam.pixel_rays(P.DEPTH_RAY_SCALE)
            self._sec = sec.astype(np.float32)
            self._h_min = float(tmap.meta["height_min_m"])
            self._h_max = float(tmap.meta["height_max_m"])
        return self._cam, self._sec, self._h_min, self._h_max

    def _elevation(self, depth_mm: np.ndarray) -> np.ndarray:
        cam, sec, h_min, h_max = self._camera()
        depth_m = DEPTH_OFFSET_M + depth_mm.astype(np.float32) / 1000.0
        z = cam.cam_height_m - depth_m / sec
        z[depth_mm == DEPTH_NO_HIT] = h_min
        return ((z - h_min) / (h_max - h_min)).astype(np.float16)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        reader = EpisodeReader(entry.ep_dir)
        window = reader.read_window(0, entry.n_frames)
        reader.close()
        field_index = {name: i for i, name in enumerate(window["state_fields"])}
        table = window["states"]

        def pick(names: list[str]) -> np.ndarray:
            return np.stack([table[:, field_index[n]] for n in names], axis=1).astype(np.float32)

        return {
            "index": index,
            "rgb": torch.from_numpy(window["rgb"]),
            "elev": torch.from_numpy(self._elevation(window["depth_mm"])),
            "z1": torch.from_numpy(pick(STATE_FIELDS)),
            "act": torch.from_numpy(pick(DEFAULT_ACTION_FIELDS)),
            "pose": torch.from_numpy(pick(DEFAULT_ROLLOUT_FIELDS)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=["artifacts/traverse"])
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--ckpt", required=True, help="WP1 ckpt_warmup.pt (frozen encoder)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--n-q", type=int, default=8)
    parser.add_argument("--pos-enc", action="store_true")
    parser.add_argument("--slot-z", action="store_true")
    parser.add_argument("--batch", type=int, default=100, help="frames per encoder forward")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="encode only the first N episodes (smoke)")
    parser.add_argument("--zero-rgb", action="store_true",
                        help="plan 8.3 input ablation: depth-only z2 (RGB channels zeroed)")
    parser.add_argument("--zero-depth", action="store_true",
                        help="plan 8.3 input ablation: RGB-only z2 (elevation channel zeroed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.episode_manifest).read_text())
    roots = [Path(r) for r in args.roots]
    if len(roots) == 1 and not (roots[0] / "meta.json").exists():
        roots = sorted(p for p in roots[0].iterdir() if p.is_dir() and any(p.glob("ep_*/meta.json")))
    entries = P._episode_entries(roots, manifest)
    if args.limit:
        entries = entries[: args.limit]
    print(f"encoding {len(entries)} episodes -> {out_dir}", flush=True)

    encoder = P.Encoder(
        z_dim=args.z_dim, n_q=args.n_q, pos_enc=args.pos_enc, slot_z=args.slot_z
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(state["encoder"], strict=True)
    encoder.eval()

    loader = DataLoader(
        EpisodeFrames(entries, Path(args.arena)),
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=2 if args.workers else None,
    )

    written, frames_done, t_start = 0, 0, time.time()
    for item in loader:
        entry = entries[int(item["index"])]
        key = episode_key(entry.ep_dir)
        target = out_dir / f"{key}.npz"
        rgb = item["rgb"].to(device, non_blocking=True)
        elev = item["elev"].to(device, non_blocking=True)
        chunks = []
        with torch.no_grad():
            for start in range(0, rgb.shape[0], args.batch):
                r = rgb[start : start + args.batch].permute(0, 3, 1, 2).float().div_(255.0)
                e = elev[start : start + args.batch].unsqueeze(1).float()
                if args.zero_rgb:
                    r = torch.zeros_like(r)
                if args.zero_depth:
                    e = torch.zeros_like(e)
                z2, _ = encoder(torch.cat([r, e], dim=1))
                chunks.append(z2.float().cpu())
        np.savez(
            target,
            z2=torch.cat(chunks).numpy().astype(np.float32),
            z1=item["z1"].numpy(),
            act=item["act"].numpy(),
            pose=item["pose"].numpy(),
        )
        written += 1
        frames_done += int(rgb.shape[0])
        if written % 50 == 0 or written == len(entries):
            rate = frames_done / max(time.time() - t_start, 1e-6)
            eta = (len(entries) - written) * (time.time() - t_start) / written / 60.0
            print(
                f"[{written}/{len(entries)}] {rate:7.1f} frames/s  eta {eta:5.1f} min",
                flush=True,
            )

    (out_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "episodes": [episode_key(e.ep_dir) for e in entries],
                "source_manifest": args.episode_manifest,
                "encoder_ckpt": args.ckpt,
                "z_dim": args.z_dim,
                "n_q": args.n_q,
                "pos_enc": args.pos_enc,
                "slot_z": args.slot_z,
                "zero_rgb": args.zero_rgb,
                "zero_depth": args.zero_depth,
                "state_fields": STATE_FIELDS,
                "action_fields": DEFAULT_ACTION_FIELDS,
                "pose_fields": DEFAULT_ROLLOUT_FIELDS,
            },
            indent=2,
        )
    )
    print(f"done: {written} episodes, {frames_done} frames", flush=True)


if __name__ == "__main__":
    main()
