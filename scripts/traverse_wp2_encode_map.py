"""Cache one STATIC scene feature map per episode (for the ego-crop token).

Diagnosis (WP2 notes): the global pooled z2 cannot carry a height field, which
is why the encoder's elevation channel goes unused and why z2 reaches only ~39%
of the privileged-terrain ceiling. The fix is to stop pooling for the dynamics
token and instead index the encoder's SPATIAL map at the vehicle.

Two facts make this cheap: the layout is static and the camera is fixed, so one
map per episode suffices, and the vehicle can be removed using the pose we
already record.

A plain temporal median is NOT enough. Measured over 400 episodes, 27% have >=5
of 9 sampled frames with the vehicle within one vehicle length -- it parks at
route ends, settles at the start, and sits still in contact episodes -- so it
wins the median vote and gets baked into the map. That is leakage, not just an
artifact: a blob at the parking spot tells the model where the vehicle spends
its time. Instead the vehicle footprint is rasterized per frame from its known
pose and EXCLUDED from the median, so each pixel is the median of the frames in
which the vehicle was somewhere else.

The tap is the encoder backbone's stage-2 output: 64 x 64 x 64 over the 80 m
arena = 1.25 m per cell, four times finer than the 16 x 16 map WP1 probed and
close to the 1.7 m spacing of the privileged patch that nearly doubled
accuracy. Written into the existing cache as ``map`` (float16, 512 KB/episode).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nedm.traverse import perception as P
from nedm.traverse.camera import CameraModel
from nedm.traverse.masks import (VEHICLE_HGT_M, VEHICLE_LEN_M, VEHICLE_WID_M,
                                 _box_corners, _rasterize_solids)
from nedm.traverse.storage import DEPTH_NO_HIT, DEPTH_OFFSET_M, EpisodeReader
from nedm.traverse.terrain import TerrainMap

MAP_STAGE = 4  # backbone[:4] -> 64 x 64 x 64
MEDIAN_FRAMES = 16
MASK_DILATE = 3  # px, covers shadow / antialiasing around the box


class EpisodeMedian(Dataset):
    """One item = the vehicle-free median image of one episode."""

    def __init__(self, keys: list[str], traverse_root: Path, arena_dir: Path) -> None:
        self.keys = keys
        self.root = traverse_root
        self.arena = arena_dir
        self._cam: CameraModel | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def _camera(self):
        if self._cam is None:
            tmap = TerrainMap.from_dir(self.arena)
            self.tmap = tmap
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
        return (z - h_min) / (h_max - h_min)

    def _vehicle_mask(self, state: np.ndarray, fields: dict[str, int]) -> np.ndarray:
        cam, _, _, _ = self._camera()
        x, y = float(state[fields["pos_x_m"]]), float(state[fields["pos_y_m"]])
        yaw = float(state[fields["yaw_rad"]])
        gz = float(self.tmap.height(x, y))
        box = _box_corners(x, y, yaw, VEHICLE_LEN_M, VEHICLE_WID_M, gz, gz + VEHICLE_HGT_M)
        mask = _rasterize_solids([box], cam)
        if MASK_DILATE:  # cheap square dilation: shadows/antialiasing hug the box
            out = mask.copy()
            for dy in range(-MASK_DILATE, MASK_DILATE + 1):
                for dx in range(-MASK_DILATE, MASK_DILATE + 1):
                    out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            mask = out
        return mask

    def __getitem__(self, index: int):
        key = self.keys[index]
        store, episode = key.split("__", 1)
        reader = EpisodeReader(self.root / store / episode)
        picks = np.linspace(0, reader.frames - 1, MEDIAN_FRAMES).astype(int)
        fields = {n: i for i, n in enumerate(reader.states()[0])}
        rgb, elev, masks = [], [], []
        for t in picks:
            win = reader.read_window(int(t), 1)
            rgb.append(win["rgb"][0].astype(np.float32) / 255.0)
            elev.append(self._elevation(win["depth_mm"][0]))
            masks.append(self._vehicle_mask(win["states"][0], fields))
        reader.close()

        keep = ~np.stack(masks)  # (F, H, W) True where the vehicle is absent
        stack_rgb = np.stack(rgb)
        stack_elev = np.stack(elev)
        # Median over the frames where the vehicle was elsewhere. Pixels the
        # vehicle never vacated (it never moved at all) fall back to the plain
        # median and are counted.
        never = ~keep.any(0)
        w = np.where(keep[..., None], stack_rgb, np.nan)
        med_rgb = np.nanmedian(w, axis=0)
        med_rgb[never] = np.median(stack_rgb, axis=0)[never]
        we = np.where(keep, stack_elev, np.nan)
        med_elev = np.nanmedian(we, axis=0)
        med_elev[never] = np.median(stack_elev, axis=0)[never]
        return {"index": index, "unmasked_px": int(never.sum()),
                "input": torch.from_numpy(
                    np.concatenate([med_rgb.transpose(2, 0, 1), med_elev[None]]).astype(np.float32))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--traverse-root", default="artifacts/traverse")
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--n-q", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--map-key", default="map", help="cache key to write (keeps older maps)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(args.cache)
    keys = [p.stem for p in sorted(cache.glob("*.npz"))]
    if args.limit:
        keys = keys[: args.limit]
    print(f"encoding static maps for {len(keys)} episodes", flush=True)

    encoder = P.Encoder(z_dim=args.z_dim, n_q=args.n_q).to(device)
    encoder.load_state_dict(
        torch.load(args.ckpt, map_location=device, weights_only=False)["encoder"], strict=True
    )
    encoder.eval()
    stem = encoder.backbone[:MAP_STAGE]

    loader = DataLoader(EpisodeMedian(keys, Path(args.traverse_root), Path(args.arena)),
                        batch_size=args.batch, num_workers=args.workers,
                        prefetch_factor=2 if args.workers else None)
    done, never_px, start = 0, 0, time.time()
    for item in loader:
        with torch.no_grad():
            feats = stem(item["input"].to(device)).float().cpu().numpy().astype(np.float16)
        for row, index in enumerate(item["index"].tolist()):
            target = cache / f"{keys[index]}.npz"
            with np.load(target) as data:
                payload = {k: data[k] for k in data.files}
            payload[args.map_key] = feats[row]
            np.savez(target, **payload)
        done += len(item["index"])
        never_px += int(item["unmasked_px"].sum())
        if done % 500 < args.batch:
            eta = (len(keys) - done) * (time.time() - start) / max(done, 1) / 60.0
            print(f"  [{done}/{len(keys)}] eta {eta:5.1f} min", flush=True)
    print(f"done: {done} maps, shape {feats.shape[1:]}; "
          f"pixels the vehicle never vacated: {never_px} "
          f"({never_px / max(done, 1):.1f}/episode of 65536)", flush=True)


if __name__ == "__main__":
    main()
